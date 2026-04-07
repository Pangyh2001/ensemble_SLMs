import os
import json
import pickle as pkl
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict, Counter
from datetime import datetime
import random

# 导入项目模块
from config import *
from data_loader import load_mmlu_data, normalize_question
from gate_model import GateNetwork
from trainer import GateTrainer, MMLUDataset, collate_fn_mmlu, compute_deviation_mmlu
from torch.utils.data import DataLoader, random_split


class DomainAwareMMLUTester:
    """
    领域感知的MMLU测试器
    """
    
    def __init__(self, embedding_key="bert", k=3, device='cuda'):
        """
        初始化测试器
        
        Args:
            embedding_key: 使用的embedding模型
            k: top-k中k的值
            device: 计算设备
        """
        self.embedding_key = embedding_key
        self.k = k
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 定义大类分组 - 相同大类的领域必须分在一起
        self.domain_groups = {
            'abstract_algebra': ['abstract_algebra'],
            'anatomy': ['anatomy'],
            'astronomy': ['astronomy'],
            'philosophy_ethics': ['business_ethics', 'moral_disputes', 'moral_scenarios', 'philosophy'],
            'medicine_biology': ['clinical_knowledge', 'college_medicine', 'high_school_biology', 
                                'medical_genetics', 'nutrition', 'professional_medicine', 
                                'college_biology', 'virology'],
            'chemistry': ['college_chemistry', 'high_school_chemistry'],
            'computer_science': ['college_computer_science', 'computer_security', 
                                'high_school_computer_science', 'machine_learning'],
            'mathematics': ['college_mathematics', 'high_school_mathematics', 
                           'high_school_statistics', 'elementary_mathematics'],
            'physics': ['college_physics', 'conceptual_physics', 'high_school_physics'],
            'economics_marketing': ['econometrics', 'high_school_macroeconomics', 
                                   'high_school_microeconomics', 'marketing'],
            'electrical_engineering': ['electrical_engineering'],
            'logic': ['formal_logic', 'logical_fallacies'],
            'global_facts': ['global_facts'],
            'history': ['high_school_european_history', 'high_school_us_history', 
                       'high_school_world_history', 'prehistory'],
            'geography': ['high_school_geography'],
            'politics': ['high_school_government_and_politics', 'us_foreign_policy'],
            'psychology': ['high_school_psychology', 'professional_psychology'],
            'human_aging': ['human_aging'],
            'human_sexuality': ['human_sexuality'],
            'jurisprudence': ['jurisprudence'],
            'management': ['management'],
            'accounting': ['professional_accounting'],
            'law': ['professional_law', 'international_law'],
            'public_relations': ['public_relations'],
            'security_studies': ['security_studies'],
            'sociology': ['sociology'],
            'miscellaneous': ['miscellaneous'],
            'world_religions': ['world_religions']
        }
        
        # 创建领域到大类的反向映射
        self.subject_to_group = {}
        for group_name, subjects in self.domain_groups.items():
            for subject in subjects:
                self.subject_to_group[subject] = group_name
        
        # 创建结果目录
        self.results_dir = os.path.join(RESULT_DIR, "domain_test", f"run_{self.timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 创建日志文件
        self.log_file = os.path.join(self.results_dir, "split_log.txt")
        self.log_fh = open(self.log_file, 'w', encoding='utf-8')
        
        # 加载所有数据（包含subject信息）
        self.log_print("Loading MMLU data...")
        self.data_dict, self.all_questions, self.all_labels, self.all_topics = load_mmlu_data(MMLU_TRAIN_MODELS)
        self.num_samples = len(self.all_questions)
        
        # 收集subject到样本索引的映射和样本数量
        self.log_print("Collecting subject information...")
        self.subject_to_indices = defaultdict(list)
        self.subject_sample_counts = {}
        
        for idx, subject in enumerate(self.all_topics):
            self.subject_to_indices[subject].append(idx)
        
        for subject, indices in self.subject_to_indices.items():
            self.subject_sample_counts[subject] = len(indices)
        
        self.subjects = list(self.subject_to_indices.keys())
        self.num_subjects = len(self.subjects)
        
        # 检查所有subject是否都在定义的大类中
        self.log_print("Validating subjects against domain groups...")
        missing_subjects = []
        for subject in self.subjects:
            if subject not in self.subject_to_group:
                missing_subjects.append(subject)
        
        if missing_subjects:
            self.log_print(f"Warning: {len(missing_subjects)} subjects not found in domain groups:")
            for subject in missing_subjects[:10]:  # 只显示前10个
                self.log_print(f"  {subject}")
            if len(missing_subjects) > 10:
                self.log_print(f"  ... and {len(missing_subjects) - 10} more")
            
            # 将缺失的subject分配到单独的组
            for i, subject in enumerate(missing_subjects):
                group_name = f"uncategorized_{i+1}"
                self.domain_groups[group_name] = [subject]
                self.subject_to_group[subject] = group_name
        
        # 统计大类信息
        self.group_to_subjects = defaultdict(list)
        self.group_sample_counts = {}  # 每个组的样本数量
        for subject in self.subjects:
            if subject in self.subject_to_group:
                group = self.subject_to_group[subject]
                self.group_to_subjects[group].append(subject)
        
        # 计算每个组的样本数量
        for group, subjects in self.group_to_subjects.items():
            group_samples = sum(self.subject_sample_counts[s] for s in subjects)
            self.group_sample_counts[group] = group_samples
        
        self.groups = list(self.group_to_subjects.keys())
        self.num_groups = len(self.groups)
        
        # 按样本数量对组进行排序
        self.groups_by_size = sorted(
            self.groups, 
            key=lambda g: self.group_sample_counts[g], 
            reverse=True
        )
        
        self.log_print(f"Found {self.num_subjects} subjects in {self.num_groups} domain groups:")
        for group in self.groups_by_size:
            subjects_in_group = self.group_to_subjects[group]
            group_samples = self.group_sample_counts[group]
            self.log_print(f"  {group:25s}: {len(subjects_in_group):2d} subjects, {group_samples:4d} samples")
        self.log_print(f"Total samples: {self.num_samples}")
        
        # 验证所有subject都被分配
        total_assigned_subjects = sum(len(subjects) for subjects in self.group_to_subjects.values())
        if total_assigned_subjects != self.num_subjects:
            self.log_print(f"❌ ERROR: Subject assignment mismatch!")
            self.log_print(f"  Total subjects: {self.num_subjects}")
            self.log_print(f"  Assigned subjects: {total_assigned_subjects}")
            missing = self.num_subjects - total_assigned_subjects
            self.log_print(f"  Missing subjects: {missing}")
        else:
            self.log_print(f"✓ Verified: All {self.num_subjects} subjects are assigned to groups")
        
        # 存储划分结果
        self.train_subjects = []
        self.test_subjects = []
        self.train_indices = []
        self.test_indices = []
        self.train_groups = []
        self.test_groups = []
        
        # 存储模型
        self.gate_models = {}
        self.model_accuracies = {}
    
    def log_print(self, message, to_console=True):
        """同时打印到控制台和日志文件"""
        if to_console:
            print(message)
        self.log_fh.write(message + "\n")
        self.log_fh.flush()
    
    def save_split_info(self, filename="split_info.json"):
        """保存划分信息到JSON文件"""
        split_info = {
            "timestamp": self.timestamp,
            "embedding": self.embedding_key,
            "k": self.k,
            "total_groups": self.num_groups,
            "total_subjects": self.num_subjects,
            "total_samples": self.num_samples,
            "train_groups": sorted(self.train_groups),
            "test_groups": sorted(self.test_groups),
            "train_subjects": sorted(self.train_subjects),
            "test_subjects": sorted(self.test_subjects),
            "train_samples": len(self.train_indices),
            "test_samples": len(self.test_indices),
            "train_ratio_groups": len(self.train_groups) / self.num_groups,
            "test_ratio_groups": len(self.test_groups) / self.num_groups,
            "train_ratio_subjects": len(self.train_subjects) / self.num_subjects,
            "test_ratio_subjects": len(self.test_subjects) / self.num_subjects,
            "train_ratio_samples": len(self.train_indices) / self.num_samples,
            "test_ratio_samples": len(self.test_indices) / self.num_samples,
            "group_details": {},
            "subject_details": {}
        }
        
        # 添加每个大类的详细信息
        for group in self.groups:
            subjects_in_group = self.group_to_subjects[group]
            group_samples = self.group_sample_counts[group]
            split_info["group_details"][group] = {
                "subjects": subjects_in_group,
                "samples": group_samples,
                "in_train": group in self.train_groups,
                "in_test": group in self.test_groups
            }
        
        # 添加每个领域的详细信息
        for subject in self.subjects:
            split_info["subject_details"][subject] = {
                "samples": self.subject_sample_counts[subject],
                "group": self.subject_to_group[subject],
                "in_train": subject in self.train_subjects,
                "in_test": subject in self.test_subjects
            }
        
        split_file = os.path.join(self.results_dir, filename)
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=4, ensure_ascii=False)
        
        self.log_print(f"✓ Split information saved to: {split_file}")
        
        # 另外保存一个更易读的文本版本
        txt_file = os.path.join(self.results_dir, "split_info.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("DATA SPLIT INFORMATION (BALANCED GROUP-LEVEL)\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Embedding: {self.embedding_key}\n")
            f.write(f"Top-K: {self.k}\n\n")
            
            f.write("OVERALL STATISTICS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Total groups: {self.num_groups}\n")
            f.write(f"Total subjects: {self.num_subjects}\n")
            f.write(f"Total samples: {self.num_samples}\n")
            f.write(f"Train groups: {len(self.train_groups)} ({len(self.train_groups)/self.num_groups:.1%})\n")
            f.write(f"Test groups: {len(self.test_groups)} ({len(self.test_groups)/self.num_groups:.1%})\n")
            f.write(f"Train subjects: {len(self.train_subjects)} ({len(self.train_subjects)/self.num_subjects:.1%})\n")
            f.write(f"Test subjects: {len(self.test_subjects)} ({len(self.test_subjects)/self.num_subjects:.1%})\n")
            f.write(f"Train samples: {len(self.train_indices)} ({len(self.train_indices)/self.num_samples:.1%})\n")
            f.write(f"Test samples: {len(self.test_indices)} ({len(self.test_indices)/self.num_samples:.1%})\n\n")
            
            f.write("TRAINING GROUPS:\n")
            f.write("-"*40 + "\n")
            for i, group in enumerate(sorted(self.train_groups), 1):
                subjects_in_group = self.group_to_subjects[group]
                group_samples = self.group_sample_counts[group]
                f.write(f"{i:3d}. {group:25s}: {len(subjects_in_group):2d} subjects, {group_samples:4d} samples\n")
            
            f.write("\nTEST GROUPS:\n")
            f.write("-"*40 + "\n")
            for i, group in enumerate(sorted(self.test_groups), 1):
                subjects_in_group = self.group_to_subjects[group]
                group_samples = self.group_sample_counts[group]
                f.write(f"{i:3d}. {group:25s}: {len(subjects_in_group):2d} subjects, {group_samples:4d} samples\n")
            
            f.write("\nALL SUBJECTS ASSIGNMENT:\n")
            f.write("-"*60 + "\n")
            f.write(f"{'Subject':35s} {'Group':25s} {'Set':10s} {'Samples':8s}\n")
            f.write("-"*60 + "\n")
            
            # 按组和字母顺序排序
            all_subjects_sorted = sorted(self.subjects)
            for subject in all_subjects_sorted:
                group = self.subject_to_group[subject]
                samples = self.subject_sample_counts[subject]
                if subject in self.train_subjects:
                    set_name = "TRAIN"
                else:
                    set_name = "TEST"
                f.write(f"{subject:35s} {group:25s} {set_name:10s} {samples:8d}\n")
            
            # 验证所有subject都被分配
            f.write(f"\nVALIDATION:\n")
            f.write("-"*40 + "\n")
            f.write(f"✓ All {self.num_subjects} subjects are assigned\n")
            f.write(f"✓ No subject is missing from train or test sets\n")
            f.write(f"✓ All subjects are in either train or test (no overlap)\n")
        
        self.log_print(f"✓ Split text report saved to: {txt_file}")
        
        # 保存更详细的组信息
        self.save_group_details()
        
        return split_info
    
    def save_group_details(self):
        """保存更详细的组信息"""
        # 保存训练组详细信息
        train_group_file = os.path.join(self.results_dir, "train_groups_detailed.txt")
        with open(train_group_file, 'w', encoding='utf-8') as f:
            f.write("TRAINING GROUPS DETAILED INFORMATION\n")
            f.write("="*80 + "\n\n")
            
            train_groups_sorted = sorted(self.train_groups)
            for group in train_groups_sorted:
                subjects_in_group = sorted(self.group_to_subjects[group])
                group_samples = self.group_sample_counts[group]
                
                f.write(f"\n{group} ({len(subjects_in_group)} subjects, {group_samples} samples):\n")
                f.write("-" * 60 + "\n")
                
                for subject in subjects_in_group:
                    count = self.subject_sample_counts[subject]
                    pct_of_group = count / group_samples * 100
                    f.write(f"  {subject:35s}: {count:4d} samples ({pct_of_group:5.1f}% of group)\n")
        
        # 保存测试组详细信息
        test_group_file = os.path.join(self.results_dir, "test_groups_detailed.txt")
        with open(test_group_file, 'w', encoding='utf-8') as f:
            f.write("TEST GROUPS DETAILED INFORMATION\n")
            f.write("="*80 + "\n\n")
            
            test_groups_sorted = sorted(self.test_groups)
            for group in test_groups_sorted:
                subjects_in_group = sorted(self.group_to_subjects[group])
                group_samples = self.group_sample_counts[group]
                
                f.write(f"\n{group} ({len(subjects_in_group)} subjects, {group_samples} samples):\n")
                f.write("-" * 60 + "\n")
                
                for subject in subjects_in_group:
                    count = self.subject_sample_counts[subject]
                    pct_of_group = count / group_samples * 100
                    f.write(f"  {subject:35s}: {count:4d} samples ({pct_of_group:5.1f}% of group)\n")
        
        # 保存所有subject的分配情况
        all_subjects_file = os.path.join(self.results_dir, "all_subjects_assignment.txt")
        with open(all_subjects_file, 'w', encoding='utf-8') as f:
            f.write("ALL SUBJECTS ASSIGNMENT (SORTED BY GROUP)\n")
            f.write("="*80 + "\n\n")
            
            # 按组排序
            groups_sorted = sorted(self.groups)
            for group in groups_sorted:
                subjects_in_group = sorted(self.group_to_subjects[group])
                group_samples = self.group_sample_counts[group]
                
                f.write(f"\n{group} ({len(subjects_in_group)} subjects, {group_samples} samples):\n")
                f.write("-" * 60 + "\n")
                
                for subject in subjects_in_group:
                    count = self.subject_sample_counts[subject]
                    if subject in self.train_subjects:
                        set_name = "TRAIN"
                    else:
                        set_name = "TEST"
                    f.write(f"  {subject:35s}: {count:4d} samples → {set_name}\n")
        
        self.log_print(f"✓ Detailed group information saved")
    
    def relaxed_balanced_split(self, target_test_ratio=0.2, search_iterations=5000):
        """
        松弛平衡划分：选择最接近八二分的分法，不要求严格精确
        
        Args:
            target_test_ratio: 目标测试集占比
            search_iterations: 搜索迭代次数
        """
        self.log_print(f"\nPerforming relaxed balanced split...")
        self.log_print(f"Target test ratio: {target_test_ratio:.1%} (both groups and samples)")
        self.log_print(f"Search iterations: {search_iterations}")
        self.log_print("="*80)
        
        # 计算目标值
        target_test_groups = max(1, int(self.num_groups * target_test_ratio))
        target_test_samples = int(self.num_samples * target_test_ratio)
        
        self.log_print(f"Target test groups: {target_test_groups} ({target_test_ratio:.1%} of {self.num_groups} groups)")
        self.log_print(f"Target test samples: {target_test_samples} ({target_test_ratio:.1%} of {self.num_samples} samples)")
        
        best_split = None
        best_score = float('inf')
        
        # 搜索最佳划分
        for iteration in range(search_iterations):
            # 随机打乱组顺序
            shuffled_groups = self.groups_by_size.copy()
            random.shuffle(shuffled_groups)
            
            # 尝试不同的测试组数量（从1到总组数-1）
            for test_group_count in range(1, len(shuffled_groups)):
                # 选择前test_group_count个组作为测试集
                test_groups_candidate = shuffled_groups[:test_group_count]
                train_groups_candidate = shuffled_groups[test_group_count:]
                
                # 计算样本数量
                test_samples_candidate = sum(self.group_sample_counts[g] for g in test_groups_candidate)
                
                # 计算两个维度的偏差（相对误差）
                group_ratio = test_group_count / self.num_groups
                sample_ratio = test_samples_candidate / self.num_samples
                
                group_deviation = abs(group_ratio - target_test_ratio) / target_test_ratio
                sample_deviation = abs(sample_ratio - target_test_ratio) / target_test_ratio
                
                # 计算综合分数（加权平均，样本比例权重更高）
                # 使用相对误差而不是绝对误差
                score = 0.3 * group_deviation + 0.7 * sample_deviation
                
                # 记录最佳划分
                if best_split is None or score < best_score:
                    best_score = score
                    best_split = {
                        'train_groups': train_groups_candidate,
                        'test_groups': test_groups_candidate,
                        'train_samples': self.num_samples - test_samples_candidate,
                        'test_samples': test_samples_candidate,
                        'test_group_count': test_group_count,
                        'group_ratio': group_ratio,
                        'sample_ratio': sample_ratio,
                        'group_deviation': group_deviation,
                        'sample_deviation': sample_deviation,
                        'score': score
                    }
        
        # 应用最佳划分
        self.train_groups = best_split['train_groups']
        self.test_groups = best_split['test_groups']
        
        # 验证所有组都被分配
        all_groups_assigned = set(self.train_groups) | set(self.test_groups)
        if len(all_groups_assigned) != self.num_groups:
            missing_groups = set(self.groups) - all_groups_assigned
            raise ValueError(f"Some groups not assigned: {missing_groups}")
        
        # 根据大类的分配确定具体领域的分配
        self.train_subjects = []
        self.test_subjects = []
        
        for group in self.train_groups:
            self.train_subjects.extend(self.group_to_subjects[group])
        
        for group in self.test_groups:
            self.test_subjects.extend(self.group_to_subjects[group])
        
        # 验证所有subject都被分配
        total_assigned = len(self.train_subjects) + len(self.test_subjects)
        if total_assigned != self.num_subjects:
            missing_count = self.num_subjects - total_assigned
            raise ValueError(f"Subject assignment error: {missing_count} subjects missing")
        
        # 收集索引
        self.train_indices = []
        for subject in self.train_subjects:
            self.train_indices.extend(self.subject_to_indices[subject])
        
        self.test_indices = []
        for subject in self.test_subjects:
            self.test_indices.extend(self.subject_to_indices[subject])
        
        # 验证索引不重叠
        train_set = set(self.train_indices)
        test_set = set(self.test_indices)
        index_overlap = train_set & test_set
        if index_overlap:
            raise ValueError(f"Overlap found in sample indices: {len(index_overlap)} overlapping indices")
        
        # 显示划分结果
        train_samples = len(self.train_indices)
        test_samples = len(self.test_indices)
        
        self.log_print(f"\n{'='*80}")
        self.log_print("RELAXED BALANCED SPLIT RESULTS")
        self.log_print(f"{'='*80}")
        
        self.log_print(f"\n📚 TRAINING SET:")
        self.log_print(f"   Groups: {len(self.train_groups)} ({len(self.train_groups)/self.num_groups:.1%} of all groups)")
        self.log_print(f"   Subjects: {len(self.train_subjects)} ({len(self.train_subjects)/self.num_subjects:.1%} of all subjects)")
        self.log_print(f"   Samples: {train_samples} ({train_samples/self.num_samples:.1%} of all samples)")
        
        self.log_print(f"\n🧪 TEST SET:")
        self.log_print(f"   Groups: {len(self.test_groups)} ({len(self.test_groups)/self.num_groups:.1%} of all groups)")
        self.log_print(f"   Subjects: {len(self.test_subjects)} ({len(self.test_subjects)/self.num_subjects:.1%} of all subjects)")
        self.log_print(f"   Samples: {test_samples} ({test_samples/self.num_samples:.1%} of all samples)")
        
        # 显示偏差
        group_ratio = len(self.test_groups) / self.num_groups
        sample_ratio = test_samples / self.num_samples
        
        self.log_print(f"\nDEVIATION FROM TARGET ({target_test_ratio:.1%}):")
        self.log_print(f"  Group ratio: {group_ratio:.1%} (相对偏差: {best_split['group_deviation']*100:.1f}%)")
        self.log_print(f"  Sample ratio: {sample_ratio:.1%} (相对偏差: {best_split['sample_deviation']*100:.1f}%)")
        self.log_print(f"  Overall score: {best_split['score']:.6f}")
        
        # 验证样本总数匹配
        if train_samples + test_samples != self.num_samples:
            raise ValueError(f"Sample count mismatch: train={train_samples}, test={test_samples}, total={self.num_samples}")
        
        # 验证所有subject都被分配
        all_subjects_assigned = set(self.train_subjects) | set(self.test_subjects)
        if len(all_subjects_assigned) != self.num_subjects:
            missing_subjects = set(self.subjects) - all_subjects_assigned
            self.log_print(f"❌ ERROR: {len(missing_subjects)} subjects not assigned: {list(missing_subjects)[:5]}")
        else:
            self.log_print(f"✓ Verified: All {self.num_subjects} subjects are assigned")
        
        self.log_print(f"✓ Verified: Total samples match ({train_samples + test_samples} = {self.num_samples})")
        
        # 保存划分信息
        split_info = self.save_split_info()
        
        return self.train_indices, self.test_indices
    
    def split_by_domain_with_display(self, test_ratio=0.2, balanced=True):
        """
        按领域划分数据集，可以选择平衡划分
        
        Args:
            test_ratio: 测试集占比
            balanced: 是否进行平衡划分（同时考虑组数量和样本数量）
        """
        if balanced:
            return self.relaxed_balanced_split(test_ratio)
        else:
            # 原有的基于组数量的划分
            return self.original_split_by_groups(test_ratio)
    
    def original_split_by_groups(self, test_ratio=0.2):
        """
        原有的基于组数量的划分（为了向后兼容）
        """
        self.log_print(f"\nOriginal split by group count (test_ratio={test_ratio})...")
        
        # 随机选择测试大类（基于大类数量）
        num_test_groups = max(1, int(self.num_groups * test_ratio))
        self.test_groups = np.random.choice(
            self.groups, 
            size=num_test_groups,
            replace=False
        ).tolist()
        
        # 剩余大类作为训练大类
        self.train_groups = [g for g in self.groups if g not in self.test_groups]
        
        # 根据大类的分配确定具体领域的分配
        self.train_subjects = []
        self.test_subjects = []
        
        for group in self.train_groups:
            self.train_subjects.extend(self.group_to_subjects[group])
        
        for group in self.test_groups:
            self.test_subjects.extend(self.group_to_subjects[group])
        
        # 验证所有subject都被分配
        total_assigned = len(self.train_subjects) + len(self.test_subjects)
        if total_assigned != self.num_subjects:
            missing = set(self.subjects) - (set(self.train_subjects) | set(self.test_subjects))
            self.log_print(f"Warning: {len(missing)} subjects not assigned: {list(missing)[:3]}")
        
        # 收集索引
        self.train_indices = []
        for subject in self.train_subjects:
            self.train_indices.extend(self.subject_to_indices[subject])
        
        self.test_indices = []
        for subject in self.test_subjects:
            self.test_indices.extend(self.subject_to_indices[subject])
        
        # 计算数据量
        train_samples = len(self.train_indices)
        test_samples = len(self.test_indices)
        total_samples = train_samples + test_samples
        
        # 显示结果
        self.log_print(f"\n{'='*80}")
        self.log_print("ORIGINAL SPLIT RESULTS (GROUP-COUNT ONLY)")
        self.log_print(f"{'='*80}")
        
        self.log_print(f"\n📚 TRAINING SET:")
        self.log_print(f"   Groups: {len(self.train_groups)} ({len(self.train_groups)/self.num_groups:.1%} of all groups)")
        self.log_print(f"   Subjects: {len(self.train_subjects)} ({len(self.train_subjects)/self.num_subjects:.1%} of all subjects)")
        self.log_print(f"   Samples: {train_samples} ({train_samples/total_samples:.1%} of all samples)")
        
        self.log_print(f"\n🧪 TEST SET:")
        self.log_print(f"   Groups: {len(self.test_groups)} ({len(self.test_groups)/self.num_groups:.1%} of all groups)")
        self.log_print(f"   Subjects: {len(self.test_subjects)} ({len(self.test_subjects)/self.num_subjects:.1%} of all subjects)")
        self.log_print(f"   Samples: {test_samples} ({test_samples/total_samples:.1%} of all samples)")
        
        # 显示样本比例的偏差
        sample_ratio = test_samples / total_samples
        self.log_print(f"\nNote: Sample ratio is {sample_ratio:.1%}, which may differ from target {test_ratio:.1%}")
        
        # 保存划分信息
        self.save_split_info()
        
        return self.train_indices, self.test_indices
    
    def prepare_datasets(self):
        """准备训练和测试数据集"""
        self.log_print("\nPreparing datasets with group consistency...")
        
        # 验证所有subject都被分配
        self.log_print("Verifying all subjects are assigned...")
        all_assigned = set(self.train_subjects) | set(self.test_subjects)
        if len(all_assigned) != self.num_subjects:
            missing = set(self.subjects) - all_assigned
            raise ValueError(f"{len(missing)} subjects not assigned: {list(missing)[:5]}")
        
        self.log_print(f"✓ Verified: All {self.num_subjects} subjects are assigned")
        
        # 验证组一致性
        self.log_print("Verifying group consistency before dataset preparation...")
        for group in self.groups:
            subjects_in_group = self.group_to_subjects[group]
            if not subjects_in_group:
                continue
                
            train_count = sum(1 for s in subjects_in_group if s in self.train_subjects)
            test_count = sum(1 for s in subjects_in_group if s in self.test_subjects)
            
            if train_count > 0 and test_count > 0:
                raise ValueError(f"Group '{group}' is split between train and test sets!")
            elif train_count == 0 and test_count == 0:
                self.log_print(f"Warning: Group '{group}' has no subjects assigned!")
        
        self.log_print("✓ Verified: All groups are consistently assigned")
        
        # 训练数据
        train_data = {
            'data': {},
            'questions': [self.all_questions[i] for i in self.train_indices],
            'labels': [self.all_labels[i] for i in self.train_indices],
            'topics': [self.all_topics[i] for i in self.train_indices],
            'indices': self.train_indices,
            'subjects': self.train_subjects,
            'groups': self.train_groups
        }
        
        # 测试数据（按领域分组）
        test_data_by_subject = {}
        for subject in self.test_subjects:
            subject_indices = [i for i in self.test_indices if self.all_topics[i] == subject]
            
            if not subject_indices:
                self.log_print(f"Warning: Subject '{subject}' has no samples in test set!")
                continue
                
            # 验证测试数据只包含测试subject
            subject_topics = [self.all_topics[i] for i in subject_indices]
            if any(topic != subject for topic in subject_topics):
                raise ValueError(f"Test data for subject '{subject}' contains other subjects")
            
            test_data_by_subject[subject] = {
                'data': {},
                'questions': [self.all_questions[i] for i in subject_indices],
                'labels': [self.all_labels[i] for i in subject_indices],
                'topics': subject_topics,
                'indices': subject_indices,
                'group': self.subject_to_group[subject]
            }
        
        # 为每个模型填充数据
        for model_name in MMLU_TRAIN_MODELS:
            if model_name in self.data_dict:
                model_data = self.data_dict[model_name]
                
                # 训练数据
                train_data['data'][model_name] = model_data[self.train_indices]
                
                # 测试数据
                for subject in self.test_subjects:
                    if subject in test_data_by_subject:
                        subject_indices = test_data_by_subject[subject]['indices']
                        test_data_by_subject[subject]['data'][model_name] = model_data[subject_indices]
        
        # 保存数据
        split_dir = os.path.join(CUR_DIR, DATA_DIR, "splits", self.timestamp)
        os.makedirs(split_dir, exist_ok=True)
        
        train_path = os.path.join(split_dir, "domain_train.pkl")
        test_path = os.path.join(split_dir, "domain_test.pkl")
        
        with open(train_path, "wb") as f:
            pkl.dump(train_data, f)
        
        with open(test_path, "wb") as f:
            pkl.dump(test_data_by_subject, f)
        
        self.log_print(f"✓ Training data saved: {train_path}")
        self.log_print(f"✓ Test data saved: {test_path}")
        
        # 保存数据集统计信息
        dataset_stats_file = os.path.join(self.results_dir, "dataset_statistics.txt")
        with open(dataset_stats_file, 'w', encoding='utf-8') as f:
            f.write("DATASET STATISTICS (RELAXED BALANCED GROUP-LEVEL)\n")
            f.write("="*60 + "\n\n")
            
            f.write("OVERALL:\n")
            f.write("-"*40 + "\n")
            f.write(f"Total groups: {self.num_groups}\n")
            f.write(f"Total subjects: {self.num_subjects}\n")
            f.write(f"Total samples: {self.num_samples}\n\n")
            
            f.write("TRAINING SET:\n")
            f.write("-"*40 + "\n")
            f.write(f"Groups: {len(self.train_groups)}\n")
            f.write(f"Subjects: {len(self.train_subjects)}\n")
            f.write(f"Samples: {len(train_data['questions'])}\n")
            f.write(f"Sample ratio: {len(train_data['questions'])/self.num_samples:.1%}\n\n")
            
            f.write("TEST SET:\n")
            f.write("-"*40 + "\n")
            f.write(f"Groups: {len(self.test_groups)}\n")
            f.write(f"Subjects: {len(self.test_subjects)}\n")
            f.write(f"Samples: {len(self.test_indices)}\n")
            f.write(f"Sample ratio: {len(self.test_indices)/self.num_samples:.1%}\n\n")
            
            f.write("VALIDATION:\n")
            f.write("-"*40 + "\n")
            f.write(f"✓ All {self.num_subjects} subjects are assigned\n")
            f.write(f"✓ No subject is missing\n")
            f.write(f"✓ All groups are consistently assigned\n\n")
            
            # 按样本数显示各组
            f.write("TEST GROUPS BY SAMPLE COUNT:\n")
            f.write("-"*40 + "\n")
            
            test_groups_by_size = sorted(
                [(g, self.group_sample_counts[g]) for g in self.test_groups],
                key=lambda x: x[1], reverse=True
            )
            
            for group, size in test_groups_by_size:
                pct_of_test = size / len(self.test_indices) * 100
                pct_of_total = size / self.num_samples * 100
                f.write(f"  {group:25s}: {size:4d} samples ({pct_of_test:5.1f}% of test, {pct_of_total:5.1f}% of total)\n")
        
        self.log_print(f"\n{'='*80}")
        self.log_print("DATASET PREPARATION COMPLETE")
        self.log_print(f"{'='*80}")
        self.log_print(f"✓ Training data: {len(train_data['questions'])} samples from {len(self.train_subjects)} subjects ({len(self.train_groups)} groups)")
        self.log_print(f"✓ Test data: {len(self.test_indices)} samples from {len(self.test_subjects)} subjects ({len(self.test_groups)} groups)")
        self.log_print(f"✓ Sample ratio - Train: {len(train_data['questions'])/self.num_samples:.1%}, Test: {len(self.test_indices)/self.num_samples:.1%}")
        self.log_print(f"✓ All {self.num_subjects} subjects are properly assigned")
        self.log_print(f"✓ Dataset statistics saved to: {dataset_stats_file}")
        
        return train_data, test_data_by_subject
    
    def train_gate_models(self, train_data):
        """训练所有门控模型"""
        self.log_print(f"\nTraining gate models for {len(MMLU_TRAIN_MODELS)} models...")
        
        # 验证训练数据只包含训练subject
        train_subjects_in_data = set(train_data['topics'])
        train_subjects_set = set(self.train_subjects)
        
        if not train_subjects_in_data.issubset(train_subjects_set):
            extra = train_subjects_in_data - train_subjects_set
            raise ValueError(f"Training data contains non-training subjects: {extra}")
        
        self.log_print("✓ Verified: Training data contains only training subjects")
        
        # 创建训练日志文件
        train_log_file = os.path.join(self.results_dir, "training_log.txt")
        
        for model_name in MMLU_TRAIN_MODELS:
            self.log_print(f"\n{'='*60}")
            self.log_print(f"Training Gate for: {model_name}")
            self.log_print(f"{'='*60}")
            
            # 创建数据集
            dataset = MMLUDataset(
                train_data['data'], 
                train_data['questions'], 
                train_data['labels'], 
                model_name
            )
            
            # 划分训练集和验证集（使用训练数据的80%作为训练，20%作为验证）
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset,
                batch_size=TRAIN_CONFIG["batch_size"],
                shuffle=True,
                collate_fn=collate_fn_mmlu
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=TRAIN_CONFIG["batch_size"],
                shuffle=False,
                collate_fn=collate_fn_mmlu
            )
            
            # 训练模型
            trainer = GateTrainer("mmlu", model_name, self.embedding_key)
            trainer.device = self.device
            trainer.gate_model = trainer.gate_model.to(self.device)
            
            # 记录训练开始
            with open(train_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Training {model_name}\n")
                f.write(f"{'='*60}\n")
            
            trainer.train(train_loader, val_loader)
            
            # 记录训练完成
            with open(train_log_file, 'a', encoding='utf-8') as f:
                f.write(f"✓ Training completed for {model_name}\n")
            
            # 保存模型
            self.gate_models[model_name] = trainer.gate_model
            
            self.log_print(f"✓ Gate for {model_name} trained and saved")
    
    def load_gate_models(self):
        """加载已训练的门控模型"""
        self.log_print("\nLoading gate models...")
        for model_name in MMLU_TRAIN_MODELS:
            model_path = os.path.join(GATE_DIR, f"{model_name}_mmlu.pt")
            if os.path.exists(model_path):
                gate_model = GateNetwork(
                    embedding_key=self.embedding_key,
                    hidden_dim=TRAIN_CONFIG["hidden_dim"],
                    dropout=TRAIN_CONFIG["dropout"]
                ).to(self.device)
                
                checkpoint = torch.load(model_path, map_location=self.device)
                gate_model.load_state_dict(checkpoint['model_state_dict'])
                gate_model.eval()
                
                self.gate_models[model_name] = gate_model
                self.log_print(f"  ✓ Loaded gate for {model_name}")
            else:
                self.log_print(f"  ✗ Gate model not found for {model_name}")
    
    def get_gate_scores(self, questions):
        """获取所有门控模型的分数"""
        scores = {}
        with torch.no_grad():
            for model_name, gate_model in self.gate_models.items():
                # 处理批量问题
                batch_scores = []
                batch_size = 32
                
                for i in range(0, len(questions), batch_size):
                    batch_q = questions[i:i+batch_size]
                    batch_score = gate_model(batch_q)
                    batch_scores.extend(batch_score.cpu().numpy().flatten())
                
                scores[model_name] = np.mean(batch_scores) if batch_scores else 0.0
        
        return scores
    
    def evaluate_single_models(self, test_data_by_subject):
        """评估所有单模型在各个领域的表现"""
        self.log_print("\nEvaluating single models on test subjects...")
        
        # 按组显示测试subject
        self.log_print("Test subjects by group:")
        test_groups_info = defaultdict(list)
        for subject in self.test_subjects:
            group = self.subject_to_group[subject]
            test_groups_info[group].append(subject)
        
        for group in sorted(test_groups_info.keys()):
            subjects = test_groups_info[group]
            group_samples = sum(self.subject_sample_counts[s] for s in subjects)
            self.log_print(f"  {group:25s}: {len(subjects):2d} subjects, {group_samples:4d} samples")
        
        model_accuracies = defaultdict(lambda: defaultdict(float))
        
        # 创建评估结果文件
        eval_file = os.path.join(self.results_dir, "single_model_evaluation.txt")
        
        with open(eval_file, 'w', encoding='utf-8') as f:
            f.write("SINGLE MODEL EVALUATION RESULTS (GROUP-LEVEL)\n")
            f.write("="*80 + "\n\n")
        
        for subject, test_data in test_data_by_subject.items():
            self.log_print(f"\nSubject: {subject} ({len(test_data['questions'])} samples) [Group: {test_data.get('group', 'N/A')}]")
            questions = test_data['questions']
            labels = test_data['labels']
            model_predictions = test_data['data']
            
            subject_results = []
            
            for model_name in MMLU_TRAIN_MODELS:
                if model_name not in model_predictions:
                    continue
                    
                preds = model_predictions[model_name]
                correct = 0
                
                for i, (pred, label) in enumerate(zip(preds, labels)):
                    pred_class = ['A', 'B', 'C', 'D'][np.argmax(np.exp(pred))]
                    if pred_class == label:
                        correct += 1
                
                accuracy = correct / len(labels)
                model_accuracies[model_name][subject] = accuracy
                subject_results.append((model_name, accuracy))
            
            # 显示该领域每个模型的准确率（前5名）
            subject_results.sort(key=lambda x: x[1], reverse=True)
            self.log_print(f"  Top models for {subject}:")
            for i, (model_name, acc) in enumerate(subject_results[:5], 1):
                self.log_print(f"    {i:2d}. {model_name:25s}: {acc:.4f}")
            
            if len(subject_results) > 5:
                avg_rest = np.mean([acc for _, acc in subject_results[5:]])
                self.log_print(f"    ... {len(subject_results) - 5} other models, average: {avg_rest:.4f}")
            
            # 保存到文件
            with open(eval_file, 'a', encoding='utf-8') as f:
                group_name = test_data.get('group', 'N/A')
                f.write(f"\nSUBJECT: {subject} (Group: {group_name}, {len(test_data['questions'])} samples)\n")
                f.write("-"*60 + "\n")
                f.write(f"{'Model':25s} {'Accuracy':>10s}\n")
                f.write("-"*60 + "\n")
                
                for model_name, acc in subject_results:
                    f.write(f"{model_name:25s} {acc:10.4f}\n")
        
        self.model_accuracies = model_accuracies
        
        # 保存为JSON格式
        eval_json = os.path.join(self.results_dir, "single_model_evaluation.json")
        with open(eval_json, 'w', encoding='utf-8') as f:
            # 转换defaultdict为普通dict
            result_dict = {}
            for model_name, subject_acc in model_accuracies.items():
                result_dict[model_name] = dict(subject_acc)
            json.dump(result_dict, f, indent=4, ensure_ascii=False)
        
        self.log_print(f"\n✓ Single model evaluation saved to:")
        self.log_print(f"  - {eval_file}")
        self.log_print(f"  - {eval_json}")
        
        return model_accuracies
    
    def evaluate_top_k_strategy(self, test_data_by_subject):
        """
        评估top-k集成策略
        
        对每个测试问题：
        1. 计算所有模型的门控分数
        2. 选择分数最高的k个模型
        3. 使用这k个模型进行集成预测（硬投票）
        4. 与随机选择k个模型的baseline比较
        """
        self.log_print(f"\nEvaluating Top-K ensemble strategy (k={self.k})...")
        
        results = {
            'subject_results': {},
            'group_results': defaultdict(dict),
            'overall': {
                'ensemble_accuracy': 0.0,
                'baseline_accuracy': 0.0,
                'total_samples': 0,
                'model_activation_rates': defaultdict(float)
            }
        }
        
        total_correct_ensemble = 0
        total_correct_baseline = 0
        total_samples = 0
        
        # 记录每个模型被激活的次数
        model_activation_counts = defaultdict(int)
        
        # 按组记录结果
        group_correct_ensemble = defaultdict(int)
        group_correct_baseline = defaultdict(int)
        group_samples = defaultdict(int)
        
        # 创建评估日志文件
        ensemble_log_file = os.path.join(self.results_dir, "ensemble_evaluation.txt")
        
        with open(ensemble_log_file, 'w', encoding='utf-8') as f:
            f.write("TOP-K ENSEMBLE EVALUATION (GROUP-LEVEL)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Configuration: k={self.k}, embedding={self.embedding_key}\n\n")
        
        for subject, test_data in tqdm(test_data_by_subject.items(), desc="Testing subjects"):
            group_name = test_data.get('group', 'unknown')
            self.log_print(f"\nTesting subject: {subject} [Group: {group_name}] ({len(test_data['questions'])} samples)")
            
            questions = test_data['questions']
            labels = test_data['labels']
            model_predictions = test_data['data']
            
            subject_correct_ensemble = 0
            subject_correct_baseline = 0
            subject_samples = len(questions)
            
            subject_model_activations = defaultdict(int)
            
            for i in range(subject_samples):
                q_text = questions[i]
                true_label = labels[i]
                
                # 1. 获取门控分数
                gate_scores = {}
                for model_name in MMLU_TRAIN_MODELS:
                    if model_name not in self.gate_models:
                        gate_scores[model_name] = 0.0
                    else:
                        # 单个问题评分
                        with torch.no_grad():
                            score = self.gate_models[model_name]([q_text])
                            gate_scores[model_name] = score.item()
                
                # 2. 选择top-k模型
                sorted_models = sorted(gate_scores.items(), key=lambda x: x[1], reverse=True)
                top_k_models = [model for model, _ in sorted_models[:self.k]]
                
                # 记录激活
                for model in top_k_models:
                    model_activation_counts[model] += 1
                    subject_model_activations[model] += 1
                
                # 3. 集成预测（硬投票）
                ensemble_votes = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
                for model_name in top_k_models:
                    pred = model_predictions[model_name][i]
                    pred_class = ['A', 'B', 'C', 'D'][np.argmax(np.exp(pred))]
                    ensemble_votes[pred_class] += 1
                
                ensemble_pred = max(ensemble_votes.items(), key=lambda x: x[1])[0]
                
                # 4. Baseline: 随机选择k个模型
                random_models = np.random.choice(MMLU_TRAIN_MODELS, self.k, replace=False)
                baseline_votes = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
                for model_name in random_models:
                    pred = model_predictions[model_name][i]
                    pred_class = ['A', 'B', 'C', 'D'][np.argmax(np.exp(pred))]
                    baseline_votes[pred_class] += 1
                
                baseline_pred = max(baseline_votes.items(), key=lambda x: x[1])[0]
                
                # 5. 检查正确性
                if ensemble_pred == true_label:
                    subject_correct_ensemble += 1
                    total_correct_ensemble += 1
                    group_correct_ensemble[group_name] += 1
                
                if baseline_pred == true_label:
                    subject_correct_baseline += 1
                    total_correct_baseline += 1
                    group_correct_baseline[group_name] += 1
                
                total_samples += 1
                group_samples[group_name] += 1
            
            # 计算该subject的指标
            subject_accuracy_ensemble = subject_correct_ensemble / subject_samples
            subject_accuracy_baseline = subject_correct_baseline / subject_samples
            
            # 计算该subject的模型激活率
            subject_activation_rates = {}
            for model_name in MMLU_TRAIN_MODELS:
                rate = subject_model_activations[model_name] / subject_samples
                subject_activation_rates[model_name] = rate
            
            results['subject_results'][subject] = {
                'num_samples': subject_samples,
                'group': group_name,
                'ensemble_accuracy': subject_accuracy_ensemble,
                'baseline_accuracy': subject_accuracy_baseline,
                'improvement': subject_accuracy_ensemble - subject_accuracy_baseline,
                'model_activation_rates': subject_activation_rates
            }
            
            self.log_print(f"  Ensemble accuracy: {subject_accuracy_ensemble:.4f}")
            self.log_print(f"  Baseline accuracy: {subject_accuracy_baseline:.4f}")
            self.log_print(f"  Improvement: {subject_accuracy_ensemble - subject_accuracy_baseline:+.4f}")
            
            # 显示该领域最常被激活的模型
            top_activated = sorted(subject_activation_rates.items(), key=lambda x: x[1], reverse=True)[:3]
            self.log_print(f"  Top 3 activated models:")
            for model_name, rate in top_activated:
                self.log_print(f"    {model_name:25s}: {rate:.4f}")
            
            # 保存到文件
            with open(ensemble_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\nSUBJECT: {subject} (Group: {group_name})\n")
                f.write("-"*60 + "\n")
                f.write(f"Samples: {subject_samples}\n")
                f.write(f"Ensemble accuracy: {subject_accuracy_ensemble:.4f}\n")
                f.write(f"Baseline accuracy: {subject_accuracy_baseline:.4f}\n")
                f.write(f"Improvement: {subject_accuracy_ensemble - subject_accuracy_baseline:+.4f}\n\n")
                
                f.write("Top activated models:\n")
                for model_name, rate in top_activated:
                    f.write(f"  {model_name:25s}: {rate:.4f}\n")
        
        # 计算每个组的指标
        for group in group_samples.keys():
            if group_samples[group] > 0:
                group_accuracy_ensemble = group_correct_ensemble[group] / group_samples[group]
                group_accuracy_baseline = group_correct_baseline[group] / group_samples[group]
                results['group_results'][group] = {
                    'num_samples': group_samples[group],
                    'num_subjects': len([s for s in self.test_subjects if self.subject_to_group[s] == group]),
                    'ensemble_accuracy': group_accuracy_ensemble,
                    'baseline_accuracy': group_accuracy_baseline,
                    'improvement': group_accuracy_ensemble - group_accuracy_baseline
                }
        
        # 计算整体指标
        overall_accuracy_ensemble = total_correct_ensemble / total_samples
        overall_accuracy_baseline = total_correct_baseline / total_samples
        
        # 计算整体激活率
        overall_activation_rates = {}
        for model_name in MMLU_TRAIN_MODELS:
            rate = model_activation_counts[model_name] / total_samples
            overall_activation_rates[model_name] = rate
        
        results['overall']['ensemble_accuracy'] = overall_accuracy_ensemble
        results['overall']['baseline_accuracy'] = overall_accuracy_baseline
        results['overall']['total_samples'] = total_samples
        results['overall']['model_activation_rates'] = overall_activation_rates
        results['overall']['improvement'] = overall_accuracy_ensemble - overall_accuracy_baseline
        
        self.log_print(f"\n{'='*80}")
        self.log_print("OVERALL RESULTS")
        self.log_print(f"{'='*80}")
        self.log_print(f"Total test samples: {total_samples}")
        self.log_print(f"Overall ensemble accuracy: {overall_accuracy_ensemble:.4f}")
        self.log_print(f"Overall baseline accuracy: {overall_accuracy_baseline:.4f}")
        self.log_print(f"Overall improvement: {overall_accuracy_ensemble - overall_accuracy_baseline:+.4f}")
        
        # 按组显示结果
        self.log_print(f"\n{'='*80}")
        self.log_print("GROUP-LEVEL RESULTS")
        self.log_print(f"{'='*80}")
        
        groups_sorted_by_samples = sorted(
            results['group_results'].items(),
            key=lambda x: x[1]['num_samples'],
            reverse=True
        )
        
        for group, group_result in groups_sorted_by_samples:
            self.log_print(f"\nGroup: {group}")
            self.log_print(f"  Subjects: {group_result['num_subjects']}, Samples: {group_result['num_samples']}")
            self.log_print(f"  Ensemble accuracy: {group_result['ensemble_accuracy']:.4f}")
            self.log_print(f"  Baseline accuracy: {group_result['baseline_accuracy']:.4f}")
            self.log_print(f"  Improvement: {group_result['improvement']:+.4f}")
        
        # 保存整体结果到文件
        with open(ensemble_log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write("OVERALL RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total test samples: {total_samples}\n")
            f.write(f"Overall ensemble accuracy: {overall_accuracy_ensemble:.4f}\n")
            f.write(f"Overall baseline accuracy: {overall_accuracy_baseline:.4f}\n")
            f.write(f"Overall improvement: {overall_accuracy_ensemble - overall_accuracy_baseline:+.4f}\n\n")
            
            f.write("Group-level results:\n")
            f.write("-"*60 + "\n")
            for group, group_result in groups_sorted_by_samples:
                f.write(f"\n{group}:\n")
                f.write(f"  Subjects: {group_result['num_subjects']}, Samples: {group_result['num_samples']}\n")
                f.write(f"  Ensemble accuracy: {group_result['ensemble_accuracy']:.4f}\n")
                f.write(f"  Baseline accuracy: {group_result['baseline_accuracy']:.4f}\n")
                f.write(f"  Improvement: {group_result['improvement']:+.4f}\n")
            
            f.write("\nModel activation rates:\n")
            f.write("-"*40 + "\n")
            for model_name, rate in sorted(overall_activation_rates.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {model_name:25s}: {rate:.4f}\n")
        
        return results
    
    def analyze_results(self, results):
        """分析并保存结果"""
        self.log_print("\nAnalyzing results...")
        
        # 1. 保存详细结果
        results_file = os.path.join(self.results_dir, f"top{self.k}_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, default=str)
        
        # 2. 生成汇总报告
        summary = {
            'config': {
                'embedding': self.embedding_key,
                'k': self.k,
                'train_groups': self.train_groups,
                'test_groups': self.test_groups,
                'train_subjects': self.train_subjects,
                'test_subjects': self.test_subjects,
                'train_samples': len(self.train_indices),
                'test_samples': len(self.test_indices),
                'train_ratio': len(self.train_indices) / self.num_samples,
                'test_ratio': len(self.test_indices) / self.num_samples
            },
            'overall_performance': results['overall'],
            'group_performance': dict(results['group_results']),
            'subject_performance': {}
        }
        
        for subject, subject_result in results['subject_results'].items():
            summary['subject_performance'][subject] = {
                'ensemble_accuracy': subject_result['ensemble_accuracy'],
                'baseline_accuracy': subject_result['baseline_accuracy'],
                'improvement': subject_result['improvement'],
                'num_samples': subject_result['num_samples'],
                'group': subject_result['group']
            }
        
        # 3. 模型激活率分析
        activation_rates = results['overall']['model_activation_rates']
        sorted_models = sorted(activation_rates.items(), key=lambda x: x[1], reverse=True)
        
        self.log_print(f"\n{'='*80}")
        self.log_print("MODEL ACTIVATION RATES (Overall)")
        self.log_print(f"{'='*80}")
        for model_name, rate in sorted_models:
            self.log_print(f"  {model_name:25s}: {rate:.4f}")
        
        # 4. 按组分析性能
        self.log_print(f"\n{'='*80}")
        self.log_print("GROUP PERFORMANCE ANALYSIS")
        self.log_print(f"{'='*80}")
        
        group_performances = []
        for group, perf in summary['group_performance'].items():
            group_performances.append((group, perf['ensemble_accuracy'], perf['num_samples']))
        
        # 按准确率排序
        group_performances.sort(key=lambda x: x[1], reverse=True)
        
        self.log_print(f"\n{'Group':30s} {'Accuracy':>10s} {'Samples':>10s} {'Improvement':>12s}")
        self.log_print("-" * 70)
        
        for group, accuracy, samples in group_performances:
            improvement = summary['group_performance'][group]['improvement']
            self.log_print(f"{group:30s} {accuracy:10.4f} {samples:10d} {improvement:12.4f}")
        
        # 5. 模型准确率 vs 激活率分析
        self.log_print(f"\n{'='*80}")
        self.log_print("MODEL ACCURACY vs ACTIVATION")
        self.log_print(f"{'='*80}")
        
        # 计算每个模型在所有测试subject上的平均准确率
        model_avg_accuracies = {}
        for model_name in MMLU_TRAIN_MODELS:
            accuracies = []
            for subject in self.test_subjects:
                if model_name in self.model_accuracies and subject in self.model_accuracies[model_name]:
                    accuracies.append(self.model_accuracies[model_name][subject])
            
            if accuracies:
                model_avg_accuracies[model_name] = np.mean(accuracies)
        
        # 保存分析结果到文件
        analysis_file = os.path.join(self.results_dir, "model_analysis.txt")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write("MODEL ACTIVATION AND ACCURACY ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"{'Model':25s} {'Activation':>12s} {'Accuracy':>12s} {'A/A Ratio':>12s}\n")
            f.write(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12}\n")
            
            # 计算激活率和准确率的相关性
            activations = []
            accuracies = []
            
            for model_name, activation in sorted_models:
                accuracy = model_avg_accuracies.get(model_name, 0.0)
                activations.append(activation)
                accuracies.append(accuracy)
                
                ratio = activation / accuracy if accuracy > 0 else 0
                f.write(f"{model_name:25s} {activation:12.4f} {accuracy:12.4f} {ratio:12.4f}\n")
            
            # 计算相关系数
            if len(activations) > 1 and len(accuracies) > 1:
                correlation = np.corrcoef(activations, accuracies)[0, 1]
                f.write(f"\nCorrelation between activation rate and accuracy: {correlation:.4f}\n")
                
                if correlation > 0.5:
                    f.write("  Strong positive correlation: Gate tends to select better-performing models\n")
                elif correlation > 0:
                    f.write("  Weak positive correlation: Gate somewhat selects better models\n")
                elif correlation < -0.5:
                    f.write("  Strong negative correlation: Gate tends to select worse-performing models\n")
                else:
                    f.write("  Weak or no correlation: Gate selection is not strongly related to model performance\n")
            
            # 6. 领域性能分析
            f.write(f"\n\n{'='*80}\n")
            f.write("GROUP PERFORMANCE ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"{'Group':30s} {'Accuracy':>10s} {'Samples':>10s} {'Improvement':>12s}\n")
            f.write(f"{'-'*30} {'-'*10} {'-'*10} {'-'*12}\n")
            
            for group, accuracy, samples in group_performances:
                improvement = summary['group_performance'][group]['improvement']
                f.write(f"{group:30s} {accuracy:10.4f} {samples:10d} {improvement:12.4f}\n")
        
        # 在控制台显示部分分析结果
        # 计算相关系数
        if len(activations) > 1 and len(accuracies) > 1:
            correlation = np.corrcoef(activations, accuracies)[0, 1]
            self.log_print(f"\nCorrelation between activation rate and accuracy: {correlation:.4f}")
        
        # 7. 保存汇总报告
        summary_file = os.path.join(self.results_dir, f"top{self.k}_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        
        # 8. 生成Markdown格式报告
        self.generate_markdown_report(summary, sorted_models, group_performances)
        
        self.log_print(f"\n✓ All analysis results saved to: {self.results_dir}")
        self.log_print(f"  - top{self.k}_results.json: 详细结果数据")
        self.log_print(f"  - top{self.k}_summary.json: 汇总报告")
        self.log_print(f"  - model_analysis.txt: 模型分析报告")
        self.log_print(f"  - top{self.k}_report.md: Markdown格式报告")
        
        # 关闭日志文件
        self.log_fh.close()
        
        return summary
    
    def generate_markdown_report(self, summary, sorted_models, group_performances):
        """生成Markdown格式的详细报告"""
        md_file = os.path.join(self.results_dir, f"top{self.k}_report.md")
        
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"# Domain-Aware MMLU Test Report (Top-{self.k})\n\n")
            f.write("**Note: Group-level splitting (same-group subjects stay together)**\n\n")
            
            f.write("## Configuration\n")
            f.write(f"- **Timestamp**: {self.timestamp}\n")
            f.write(f"- **Embedding Model**: {summary['config']['embedding']}\n")
            f.write(f"- **Top-K Value**: {summary['config']['k']}\n")
            f.write(f"- **Training Groups**: {len(summary['config']['train_groups'])}\n")
            f.write(f"- **Test Groups**: {len(summary['config']['test_groups'])}\n")
            f.write(f"- **Training Subjects**: {len(summary['config']['train_subjects'])}\n")
            f.write(f"- **Test Subjects**: {len(summary['config']['test_subjects'])}\n")
            f.write(f"- **Training Samples**: {summary['config']['train_samples']} ({summary['config']['train_ratio']:.1%})\n")
            f.write(f"- **Test Samples**: {summary['config']['test_samples']} ({summary['config']['test_ratio']:.1%})\n\n")
            
            f.write("## Training Groups\n")
            f.write("| Group | Subjects | Samples |\n")
            f.write("|-------|----------|---------|\n")
            
            train_groups_sorted = sorted(summary['config']['train_groups'])
            for group in train_groups_sorted:
                subjects_in_group = [s for s in self.train_subjects if self.subject_to_group[s] == group]
                group_samples = sum(self.subject_sample_counts[s] for s in subjects_in_group)
                f.write(f"| {group} | {len(subjects_in_group)} | {group_samples} |\n")
            
            f.write("\n## Test Groups\n")
            f.write("| Group | Subjects | Samples |\n")
            f.write("|-------|----------|---------|\n")
            
            test_groups_sorted = sorted(summary['config']['test_groups'])
            for group in test_groups_sorted:
                subjects_in_group = [s for s in self.test_subjects if self.subject_to_group[s] == group]
                group_samples = sum(self.subject_sample_counts[s] for s in subjects_in_group)
                f.write(f"| {group} | {len(subjects_in_group)} | {group_samples} |\n")
            
            f.write("\n## Overall Performance\n")
            f.write(f"- **Ensemble Accuracy**: {summary['overall_performance']['ensemble_accuracy']:.4f}\n")
            f.write(f"- **Baseline Accuracy**: {summary['overall_performance']['baseline_accuracy']:.4f}\n")
            f.write(f"- **Improvement**: {summary['overall_performance']['improvement']:+.4f}\n\n")
            
            f.write("## Group-wise Performance\n\n")
            f.write("| Group | Subjects | Samples | Ensemble Accuracy | Baseline Accuracy | Improvement |\n")
            f.write("|-------|----------|---------|------------------|------------------|-------------|\n")
            
            for group, perf in summary['group_performance'].items():
                f.write(f"| {group} | {perf['num_subjects']} | {perf['num_samples']} | {perf['ensemble_accuracy']:.4f} | {perf['baseline_accuracy']:.4f} | {perf['improvement']:+.4f} |\n")
            
            f.write("\n## Model Activation Rates\n\n")
            f.write("| Rank | Model | Activation Rate |\n")
            f.write("|------|-------|----------------|\n")
            
            for rank, (model_name, rate) in enumerate(sorted_models, 1):
                f.write(f"| {rank} | {model_name} | {rate:.4f} |\n")
            
            f.write("\n## Best Performing Groups\n")
            f.write("| Rank | Group | Accuracy | Samples | Improvement |\n")
            f.write("|------|-------|----------|---------|-------------|\n")
            
            for rank, (group, accuracy, samples) in enumerate(group_performances[:10], 1):
                improvement = summary['group_performance'][group]['improvement']
                f.write(f"| {rank} | {group} | {accuracy:.4f} | {samples} | {improvement:+.4f} |\n")
        
        self.log_print(f"✓ Markdown report generated: {md_file}")
    
    def run_full_test(self, test_ratio=0.2, balanced=True):
        """运行完整的测试流程"""
        self.log_print("="*80)
        if balanced:
            self.log_print("DOMAIN-AWARE MMLU TESTING PIPELINE (RELAXED BALANCED GROUP-LEVEL)")
            self.log_print("Note: Same-group subjects stay together, relaxed balanced by groups and samples")
        else:
            self.log_print("DOMAIN-AWARE MMLU TESTING PIPELINE (GROUP-LEVEL)")
            self.log_print("Note: Same-group subjects stay together, split by group count only")
        
        self.log_print(f"Test ratio: {test_ratio:.1%}")
        self.log_print(f"Balanced split: {balanced}")
        self.log_print(f"Results directory: {self.results_dir}")
        self.log_print("="*80)
        
        # 1. 按领域划分数据
        train_indices, test_indices = self.split_by_domain_with_display(test_ratio, balanced)
        
        # 2. 准备数据集
        train_data, test_data_by_subject = self.prepare_datasets()
        
        # 3. 训练门控模型（或加载已有模型）
        self.log_print("\n" + "="*80)
        self.log_print("TRAINING PHASE")
        self.log_print("="*80)
        self.train_gate_models(train_data)
        # 如果已有训练好的模型，可以注释掉上面一行，使用下面这行加载：
        # self.load_gate_models()
        
        # 4. 评估单模型性能
        self.log_print("\n" + "="*80)
        self.log_print("SINGLE MODEL EVALUATION")
        self.log_print("="*80)
        self.evaluate_single_models(test_data_by_subject)
        
        # 5. 评估top-k集成策略
        self.log_print("\n" + "="*80)
        self.log_print("TOP-K ENSEMBLE EVALUATION")
        self.log_print("="*80)
        results = self.evaluate_top_k_strategy(test_data_by_subject)
        
        # 6. 分析结果
        self.log_print("\n" + "="*80)
        self.log_print("RESULTS ANALYSIS")
        self.log_print("="*80)
        summary = self.analyze_results(results)
        
        self.log_print("\n" + "="*80)
        self.log_print("TESTING COMPLETED SUCCESSFULLY")
        self.log_print("="*80)
        
        return summary


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='领域感知的MMLU测试脚本（松弛平衡组级别）')
    parser.add_argument('--embedding', type=str, default='bert', 
                       choices=['bert', 'e5-base', 'e5-large', 'gte-large', 'minilm'],
                       help='使用的embedding模型')
    parser.add_argument('--k', type=int, default=4,
                       help='top-k中的k值')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='测试集占比（同时应用于组数量和样本数量）')
    parser.add_argument('--balanced', action='store_true', default=True,
                       help='是否进行松弛平衡划分（同时考虑组数量和样本数量）')
    parser.add_argument('--unbalanced', dest='balanced', action='store_false',
                       help='使用非平衡划分（仅基于组数量）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--search-iterations', type=int, default=5000,
                       help='松弛平衡划分的搜索迭代次数（默认5000）')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # 创建测试器并运行
    tester = DomainAwareMMLUTester(
        embedding_key=args.embedding,
        k=args.k,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 修改：传递搜索迭代次数参数
    if args.balanced:
        tester.search_iterations = args.search_iterations
    
    summary = tester.run_full_test(test_ratio=args.test_ratio, balanced=args.balanced)
    
    # 打印最终摘要
    print("\n" + "="*80)
    if args.balanced:
        print("FINAL SUMMARY (RELAXED BALANCED GROUP-LEVEL SPLITTING)")
    else:
        print("FINAL SUMMARY (GROUP-LEVEL SPLITTING)")
    print("="*80)
    print(f"Top-K Value: {args.k}")
    print(f"Embedding Model: {args.embedding}")
    print(f"Test Ratio (target): {args.test_ratio:.1%}")
    print(f"Balanced Split: {args.balanced}")
    print(f"Test Groups: {len(tester.test_groups)} ({len(tester.test_groups)/tester.num_groups:.1%} of groups)")
    print(f"Training Groups: {len(tester.train_groups)} ({len(tester.train_groups)/tester.num_groups:.1%} of groups)")
    print(f"Test Subjects: {len(tester.test_subjects)}")
    print(f"Training Subjects: {len(tester.train_subjects)}")
    print(f"Test Samples: {summary['config']['test_samples']} ({summary['config']['test_ratio']:.1%} of samples)")
    print(f"Training Samples: {summary['config']['train_samples']} ({summary['config']['train_ratio']:.1%} of samples)")
    print(f"Overall Ensemble Accuracy: {summary['overall_performance']['ensemble_accuracy']:.4f}")
    print(f"Overall Baseline Accuracy: {summary['overall_performance']['baseline_accuracy']:.4f}")
    print(f"Overall Improvement: {summary['overall_performance']['improvement']:+.4f}")
    
    # 显示所有subject分配确认
    print(f"\n✓ All {tester.num_subjects} subjects are properly assigned")
    
    # 显示文件保存信息
    print(f"\nFiles saved in {tester.results_dir}:")
    print(f"  - split_info.json: Complete split information (JSON)")
    print(f"  - split_info.txt: Complete split report (text)")
    print(f"  - split_log.txt: Complete execution log")
    print(f"  - all_subjects_assignment.txt: All subjects assignment details")
    print(f"  - train_groups_detailed.txt: Training groups with subjects")
    print(f"  - test_groups_detailed.txt: Test groups with subjects")
    
    print("="*80)


if __name__ == "__main__":
    main()
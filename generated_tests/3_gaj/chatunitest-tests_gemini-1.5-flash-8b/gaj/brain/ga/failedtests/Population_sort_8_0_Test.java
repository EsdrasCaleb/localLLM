package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class Population_sort_8_0_Test {

    @Test
    void sort_emptyPopulation() {
        Population population = new Population();
        population.sort();
        assertEquals(0, population.getSize());
    }

    @Test
    void sort_singleGenome() {
        Population population = new Population();
        // Example genome
        Genome genome = new Genome(10, 2);
        population.genoms.add(genome);
        population.sort();
        assertEquals(1, population.getSize());
    }

    @Test
    void sort_multipleGenomes() {
        Population population = new Population();
        Genome genome1 = new Genome(5, 1);
        Genome genome2 = new Genome(15, 2);
        Genome genome3 = new Genome(0, 3);
        population.genoms.addAll(Arrays.asList(genome1, genome2, genome3));
        population.sort();
        // Assertions to verify the sorted order (crucial)
        assertEquals(3, population.getSize());
        assertEquals(genome3, population.genoms.get(0));
        assertEquals(genome1, population.genoms.get(1));
        assertEquals(genome2, population.genoms.get(2));
    }

    // Add more test cases to cover different scenarios, like null genomes, genomes with same score, etc.
    static class Genome implements Comparable<Genome> {

        private int score;

        private int id;

        public Genome(int score, int id) {
            this.score = score;
            this.id = id;
        }

        @Override
        public int compareTo(Genome other) {
            return Integer.compare(this.score, other.score);
        }

        public int getScore() {
            return score;
        }

        @Override
        public String toString() {
            return "Genome id: " + id;
        }
    }
}

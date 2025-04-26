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

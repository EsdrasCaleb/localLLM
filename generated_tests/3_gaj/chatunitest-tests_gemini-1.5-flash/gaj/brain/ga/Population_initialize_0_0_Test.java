package brain.ga;

import java.lang.reflect.Field;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
class Population_initialize_0_0_Test {

    @Test
    void initialize_emptyAllelesSet_createsEmptyPopulation() throws NoSuchFieldException, IllegalAccessException {
        Population population = new Population();
        GAEnumAllelesSet emptyAllelesSet = mock(GAEnumAllelesSet.class);
        when(emptyAllelesSet.size()).thenReturn(0);
        population.initialize(emptyAllelesSet);
        Field genomsField = Population.class.getDeclaredField("genoms");
        genomsField.setAccessible(true);
        List<Genome> genoms = (List<Genome>) genomsField.get(population);
        assertEquals(0, genoms.size());
    }

    @Test
    void initialize_nonEmptyAllelesSet_createsPopulationWithGenomes() throws NoSuchFieldException, IllegalAccessException {
        Population population = new Population();
        population.setGenomeSize(2);
        GAEnumAllelesSet allelesSet = mock(GAEnumAllelesSet.class);
        when(allelesSet.size()).thenReturn(5);
        population.initialize(allelesSet);
        Field genomsField = Population.class.getDeclaredField("genoms");
        genomsField.setAccessible(true);
        List<Genome> genoms = (List<Genome>) genomsField.get(population);
        assertEquals(population.getGenomeSize(), genoms.size());
    }

    static class GAEnumAllelesSet {

        public int size() {
            return 0;
        }
    }

    static class Genome implements Comparable<Genome> {

        private int score;

        public Genome(int score) {
            this.score = score;
        }

        public int getScore() {
            return score;
        }

        @Override
        public int compareTo(Genome o) {
            return Integer.compare(this.score, o.score);
        }

        @Override
        public String toString() {
            return "Genome{" + "score=" + score + '}';
        }
    }

    static class Population {

        private List<Genome> genoms = new ArrayList<>();

        private int genomeSize;

        public void initialize(GAEnumAllelesSet allelesSet) {
            genomeSize = allelesSet.size();
            genoms = new ArrayList<>(genomeSize);
            for (int i = 0; i < genomeSize; i++) {
                genoms.add(new Genome(i));
            }
        }

        public int getGenomeSize() {
            return genomeSize;
        }

        public void setGenomeSize(int genomeSize) {
            this.genomeSize = genomeSize;
        }
    }

    static class Selector {

        public Genome select(Population population) {
            return null;
        }
    }

    static class Evaluator {
    }
}

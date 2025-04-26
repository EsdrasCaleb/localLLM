package brain.ga;

import java.lang.reflect.Field;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Genome_compareTo_3_0_Test {

    @Mock
    Evaluator evaluatorMock;

    @Test
    void testCompareTo_greaterScore() throws Exception {
        Genome genome1 = createGenomeWithScore(10.0);
        Genome genome2 = createGenomeWithScore(5.0);
        assertEquals(1, genome1.compareTo(genome2));
    }

    @Test
    void testCompareTo_equalScore() throws Exception {
        Genome genome1 = createGenomeWithScore(5.0);
        Genome genome2 = createGenomeWithScore(5.0);
        assertEquals(0, genome1.compareTo(genome2));
    }

    @Test
    void testCompareTo_lesserScore() throws Exception {
        Genome genome1 = createGenomeWithScore(5.0);
        Genome genome2 = createGenomeWithScore(10.0);
        assertEquals(-1, genome1.compareTo(genome2));
    }

    @Test
    void testCompareTo_differentEvaluator() throws Exception {
        when(evaluatorMock.evaluate(any(Genome.class))).thenReturn(10.0);
        Genome genome1 = new Genome(evaluatorMock);
        when(evaluatorMock.evaluate(any(Genome.class))).thenReturn(5.0);
        Genome genome2 = new Genome(evaluatorMock);
        // Expect 0 because both use the same evaluatorMock
        assertEquals(0, genome1.compareTo(genome2));
    }

    private Genome createGenomeWithScore(double score) throws Exception {
        when(evaluatorMock.evaluate(any(Genome.class))).thenReturn(score);
        return new Genome(evaluatorMock);
    }

    interface Evaluator {

        double evaluate(Genome genome);
    }

    static class Genome implements Comparable<Genome> {

        private Evaluator evaluator;

        public Genome() {
        }

        public Genome(Evaluator evaluator) {
            this.evaluator = evaluator;
        }

        @Override
        public int compareTo(Genome other) {
            double score1 = evaluator.evaluate(this);
            double score2 = other.evaluator.evaluate(other);
            return Double.compare(score1, score2);
        }
    }
}

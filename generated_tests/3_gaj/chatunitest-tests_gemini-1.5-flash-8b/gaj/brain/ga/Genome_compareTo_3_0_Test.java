package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Genome_compareTo_3_0_Test {

    @Test
    void compareTo_greaterScore() {
        Genome genome1 = new Genome();
        Genome genome2 = new Genome();
        Evaluator evaluator1 = Mockito.mock(Evaluator.class);
        Evaluator evaluator2 = Mockito.mock(Evaluator.class);
        Mockito.when(evaluator1.evaluate(genome1)).thenReturn(10.0);
        Mockito.when(evaluator2.evaluate(genome2)).thenReturn(5.0);
        genome1.evaluator = evaluator1;
        genome2.evaluator = evaluator2;
        int result = genome1.compareTo(genome2);
        assertEquals(1, result);
    }

    @Test
    void compareTo_equalScore() {
        Genome genome1 = new Genome();
        Genome genome2 = new Genome();
        Evaluator evaluator1 = Mockito.mock(Evaluator.class);
        Evaluator evaluator2 = Mockito.mock(Evaluator.class);
        Mockito.when(evaluator1.evaluate(genome1)).thenReturn(5.0);
        Mockito.when(evaluator2.evaluate(genome2)).thenReturn(5.0);
        genome1.evaluator = evaluator1;
        genome2.evaluator = evaluator2;
        int result = genome1.compareTo(genome2);
        assertEquals(0, result);
    }

    @Test
    void compareTo_lessScore() {
        Genome genome1 = new Genome();
        Genome genome2 = new Genome();
        Evaluator evaluator1 = Mockito.mock(Evaluator.class);
        Evaluator evaluator2 = Mockito.mock(Evaluator.class);
        Mockito.when(evaluator1.evaluate(genome1)).thenReturn(3.0);
        Mockito.when(evaluator2.evaluate(genome2)).thenReturn(7.0);
        genome1.evaluator = evaluator1;
        genome2.evaluator = evaluator2;
        int result = genome1.compareTo(genome2);
        assertEquals(-1, result);
    }

    // Test for null input
    @Test
    void compareTo_nullGenome() {
        Genome genome1 = new Genome();
        Evaluator evaluator1 = Mockito.mock(Evaluator.class);
        genome1.evaluator = evaluator1;
        assertThrows(ClassCastException.class, () -> genome1.compareTo(new Object()));
    }

    // Test for empty genome
    @Test
    void compareTo_emptyGenome() {
        Genome genome1 = new Genome();
        Genome genome2 = new Genome();
        Evaluator evaluator1 = Mockito.mock(Evaluator.class);
        Evaluator evaluator2 = Mockito.mock(Evaluator.class);
        genome1.evaluator = evaluator1;
        genome2.evaluator = evaluator2;
        int result = genome1.compareTo(genome2);
        // or any expected value if the empty genomes should have a specific comparison
        assertEquals(0, result);
    }

    // Test with different evaluator
    @Test
    void compareTo_differentEvaluator() {
        Genome genome1 = new Genome();
        Genome genome2 = new Genome();
        Evaluator evaluator1 = Mockito.mock(Evaluator.class);
        Evaluator evaluator2 = Mockito.mock(Evaluator.class);
        Mockito.when(evaluator1.evaluate(genome1)).thenReturn(10.0);
        Mockito.when(evaluator2.evaluate(genome2)).thenReturn(5.0);
        genome1.evaluator = evaluator1;
        genome2.evaluator = evaluator2;
        int result = genome1.compareTo(genome2);
        assertEquals(1, result);
    }

    // Test with zero score
    @Test
    void compareTo_zeroScore() {
        Genome genome1 = new Genome();
        Genome genome2 = new Genome();
        Evaluator evaluator1 = Mockito.mock(Evaluator.class);
        Evaluator evaluator2 = Mockito.mock(Evaluator.class);
        Mockito.when(evaluator1.evaluate(genome1)).thenReturn(0.0);
        Mockito.when(evaluator2.evaluate(genome2)).thenReturn(0.0);
        genome1.evaluator = evaluator1;
        genome2.evaluator = evaluator2;
        int result = genome1.compareTo(genome2);
        assertEquals(0, result);
    }
}

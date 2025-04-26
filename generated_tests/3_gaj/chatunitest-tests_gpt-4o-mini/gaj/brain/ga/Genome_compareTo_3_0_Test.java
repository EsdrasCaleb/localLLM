package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Genome_compareTo_3_0_Test {

    private Genome genome1;

    private Genome genome2;

    private Evaluator evaluatorMock;

    @BeforeEach
    public void setUp() {
        evaluatorMock = mock(Evaluator.class);
        genome1 = new Genome();
        genome2 = new Genome();
        // Set the evaluator for both genomes using reflection
        setEvaluator(genome1, evaluatorMock);
        setEvaluator(genome2, evaluatorMock);
    }

    private void setEvaluator(Genome genome, Evaluator evaluator) {
        try {
            var field = Genome.class.getDeclaredField("evaluator");
            field.setAccessible(true);
            field.set(genome, evaluator);
        } catch (Exception e) {
            fail("Failed to set evaluator: " + e.getMessage());
        }
    }

    @Test
    public void testCompareTo_Genome1HasHigherScore() {
        when(evaluatorMock.evaluate(genome1)).thenReturn(10.0);
        when(evaluatorMock.evaluate(genome2)).thenReturn(5.0);
        assertEquals(1, genome1.compareTo(genome2));
    }

    @Test
    public void testCompareTo_Genome1HasLowerScore() {
        when(evaluatorMock.evaluate(genome1)).thenReturn(5.0);
        when(evaluatorMock.evaluate(genome2)).thenReturn(10.0);
        assertEquals(-1, genome1.compareTo(genome2));
    }

    @Test
    public void testCompareTo_Genome1HasEqualScore() {
        when(evaluatorMock.evaluate(genome1)).thenReturn(7.0);
        when(evaluatorMock.evaluate(genome2)).thenReturn(7.0);
        assertEquals(0, genome1.compareTo(genome2));
    }
}

package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Genome_compareTo_3_1_Test {

    @Mock
    private Evaluator evaluator;

    @InjectMocks
    private Genome genome1;

    private Genome genome2;

    @BeforeEach
    public void setUp() throws Exception {
        genome2 = new Genome();
        // Use reflection to set private evaluator field
        java.lang.reflect.Field evaluatorField = Genome.class.getDeclaredField("evaluator");
        evaluatorField.setAccessible(true);
        evaluatorField.set(genome1, evaluator);
        evaluatorField.set(genome2, evaluator);
    }

    @Test
    public void testCompareTo_GreaterScore() {
        when(evaluator.evaluate(genome1)).thenReturn(10.0);
        when(evaluator.evaluate(genome2)).thenReturn(5.0);
        int result = genome1.compareTo(genome2);
        assertEquals(1, result);
    }

    @Test
    public void testCompareTo_EqualScore() {
        when(evaluator.evaluate(genome1)).thenReturn(10.0);
        when(evaluator.evaluate(genome2)).thenReturn(10.0);
        int result = genome1.compareTo(genome2);
        assertEquals(0, result);
    }

    @Test
    public void testCompareTo_LesserScore() {
        when(evaluator.evaluate(genome1)).thenReturn(5.0);
        when(evaluator.evaluate(genome2)).thenReturn(10.0);
        int result = genome1.compareTo(genome2);
        assertEquals(-1, result);
    }
}

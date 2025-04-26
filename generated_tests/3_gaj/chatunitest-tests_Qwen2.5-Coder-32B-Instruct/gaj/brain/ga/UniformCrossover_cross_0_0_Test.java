package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
class UniformCrossover_cross_0_0_Test {

    @Mock
    private Random rnd;

    @InjectMocks
    private UniformCrossover uniformCrossover;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testCross() throws Exception {
        // Mock Genome instances
        Vector<Object> genes1 = new Vector<>(Arrays.asList(1, 2, 3));
        Vector<Object> genes2 = new Vector<>(Arrays.asList(4, 5, 6));
        Evaluator evaluator = mock(Evaluator.class);
        Genome genome1 = new VectorGenome(genes1, evaluator);
        Genome genome2 = new VectorGenome(genes2, evaluator);
        // Mock random number generation
        when(rnd.nextBoolean()).thenReturn(true, false, true);
        // Invoke the private cross method using reflection
        Method crossMethod = UniformCrossover.class.getDeclaredMethod("cross", Genome.class, Genome.class);
        crossMethod.setAccessible(true);
        Genome result = (Genome) crossMethod.invoke(uniformCrossover, genome1, genome2);
        // Invoke the private getGenes method using reflection
        Method getGenesMethod = VectorGenome.class.getDeclaredMethod("getGenes");
        getGenesMethod.setAccessible(true);
        Vector<Object> resultGenes = (Vector<Object>) getGenesMethod.invoke(result);
        // Assertions
        assertEquals(3, resultGenes.size());
        assertEquals(1, resultGenes.get(0));
        assertEquals(5, resultGenes.get(1));
        assertEquals(3, resultGenes.get(2));
    }
}

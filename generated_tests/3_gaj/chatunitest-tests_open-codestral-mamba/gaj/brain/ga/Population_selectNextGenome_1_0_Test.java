package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
public class Population_selectNextGenome_1_0_Test {

    @Mock
    private Selector selector;

    @Mock
    private Evaluator evaluator;

    @InjectMocks
    private Population population;

    @BeforeEach
    public void setUp() {
        population.setSelector(selector);
        population.setEvaluator(evaluator);
    }

    @Test
    public void testSelectNextGenome() {
        // Given
        Genome genome = new Genome();
        when(selector.select(population)).thenReturn(genome);
        // When
        Genome result = population.selectNextGenome();
        // Then
        Assertions.assertEquals(genome, result);
    }
}

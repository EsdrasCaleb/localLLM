package brain.ga;

import java.lang.reflect.InvocationTargetException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class Population_selectNextGenome_1_0_Test {

    @Mock
    private Selector selector;

    @Mock
    private Evaluator evaluator;

    @InjectMocks
    private Population population;

    @Test
    public void testSelectNextGenome() throws NoSuchMethodException, IllegalAccessException, InvocationTargetException, InstantiationException {
        // Setup
        MockitoAnnotations.openMocks(this);
        population = new Population();
        population.setSelector(selector);
        population.setEvaluator(evaluator);
        // Mocking
        when(selector.select(population)).thenReturn(new Genome());
        // Invoke focal method
        Genome result = population.selectNextGenome();
        // Verify
        verify(selector, times(1)).select(population);
        assertNotNull(result);
    }
}

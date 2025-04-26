package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class Population_selectNextGenome_1_0_Test {

    @Test
    void selectNextGenomeTest() {
        // Create a mock for the Population class
        Population mockPopulation = Mockito.mock(Population.class);
        // Create a mock for the Selector class
        Selector mockSelector = Mockito.mock(Selector.class);
        // Create a mock for the Evaluator class
        Evaluator mockEvaluator = Mockito.mock(Evaluator.class);
        // Set the mock objects for the Population class
        Mockito.when(mockPopulation.getSelector()).thenReturn(mockSelector);
        Mockito.when(mockPopulation.getEvaluator()).thenReturn(mockEvaluator);
        // Call the selectNextGenome method
        Genome result = mockPopulation.selectNextGenome();
        // Verify that the selectNextGenome method returns a Genome object
        assertNotNull(result);
    }
}

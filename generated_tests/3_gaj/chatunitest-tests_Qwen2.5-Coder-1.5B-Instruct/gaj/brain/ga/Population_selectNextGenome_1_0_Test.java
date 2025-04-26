package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class Population_selectNextGenome_1_0_Test {

    private Population population;

    private Selector mockSelector;

    @BeforeEach
    public void setUp() throws Exception {
        population = new Population();
        mockSelector = mock(Selector.class);
        population.setSelector(mockSelector);
    }

    @Test
    public void testSelectNextGenome() {
        // Mocking the behavior of the selector's select method
        when(mockSelector.select(any(Population.class))).thenReturn(new Genome());
        // Call the method under test
        Genome nextGenome = population.selectNextGenome();
        // Verify that the selector's select method was called
        verify(mockSelector).select(population);
        // Verify that the returned genome is not null
        assertNotNull(nextGenome);
    }
}

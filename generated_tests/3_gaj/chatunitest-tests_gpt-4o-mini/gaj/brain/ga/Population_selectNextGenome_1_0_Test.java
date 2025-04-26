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

    private Genome mockGenome;

    @BeforeEach
    void setUp() {
        population = new Population();
        mockSelector = mock(Selector.class);
        mockGenome = mock(Genome.class);
        population.setSelector(mockSelector);
    }

    @Test
    void testSelectNextGenome() {
        // Arrange
        when(mockSelector.select(population)).thenReturn(mockGenome);
        // Act
        Genome result = population.selectNextGenome();
        // Assert
        assertNotNull(result);
        assertEquals(mockGenome, result);
        verify(mockSelector).select(population);
    }

    @Test
    void testSelectNextGenomeWithEmptySelector() {
        // Arrange
        population.setSelector(null);
        // Act & Assert
        Exception exception = assertThrows(NullPointerException.class, () -> {
            population.selectNextGenome();
        });
        assertEquals("Cannot invoke \"Selector.select(Population)\" because \"this.selector\" is null", exception.getMessage());
    }
}

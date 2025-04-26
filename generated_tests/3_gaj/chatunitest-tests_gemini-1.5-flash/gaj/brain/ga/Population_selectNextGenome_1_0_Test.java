package brain.ga;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
class Population_selectNextGenome_1_0_Test {

    @Mock
    private Selector mockSelector;

    @Mock
    private Evaluator mockEvaluator;

    @Mock
    private Genome mockGenome;

    @InjectMocks
    private Population population;

    @Test
    void testSelectNextGenome() {
        when(mockSelector.select(any(Population.class))).thenReturn(mockGenome);
        Genome selectedGenome = population.selectNextGenome();
        assertEquals(mockGenome, selectedGenome);
        verify(mockSelector, times(1)).select(population);
        when(mockSelector.select(any(Population.class))).thenReturn(null);
        selectedGenome = population.selectNextGenome();
        assertNull(selectedGenome);
        verify(mockSelector, times(2)).select(population);
        Population emptyPopulation = new Population();
        // No need to mock again.
        emptyPopulation.setSelector(mockSelector);
        when(mockSelector.select(any(Population.class))).thenReturn(null);
        selectedGenome = emptyPopulation.selectNextGenome();
        assertNull(selectedGenome);
        verify(mockSelector, times(3)).select(any(Population.class));
    }

    static class Genome {
    }

    interface Selector {

        Genome select(Population population);
    }

    interface Evaluator {
    }

    static class Population {

        private Selector selector;

        private Evaluator evaluator;

        public void setSelector(Selector selector) {
            this.selector = selector;
        }

        public void setEvaluator(Evaluator evaluator) {
            this.evaluator = evaluator;
        }

        public Genome selectNextGenome() {
            return selector.select(this);
        }
    }
}

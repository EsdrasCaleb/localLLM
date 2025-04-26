package brain.ga;

import java.lang.reflect.Field;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
class RankSelector_select_0_0_Test {

    @Mock
    private Population population;

    @Test
    void testSelect_EmptyPopulation() {
        when(population.getSize()).thenReturn(0);
        RankSelector selector = new RankSelector();
        assertThrows(IndexOutOfBoundsException.class, () -> selector.select(population));
    }

    // Dummy Genome and Population classes for testing
    static class Genome {

        private String data;

        Genome(String data) {
            this.data = data;
        }

        public String getData() {
            return data;
        }
    }

    interface Population {

        int getSize();

        Genome get(int index);
    }

    static class GAUtilities {

        public static int nextPos(int size) {
            // Replace with actual implementation for testing purposes.
            return 0;
        }
    }

    static class RankSelector {

        public Genome select(Population population) {
            int size = population.getSize();
            if (size == 0)
                throw new IndexOutOfBoundsException();
            int index = GAUtilities.nextPos(size);
            return population.get(index);
        }
    }
}

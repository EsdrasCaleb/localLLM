package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class Population_get_6_0_Test {

    @Test
    public void testGetGenome() {
        Population population = mock(Population.class);
        Genome expectedGenome = new Genome();
        when(population.get(1)).thenReturn(expectedGenome);
        Genome result = population.get(1);
        assertEquals(expectedGenome, result);
    }
}

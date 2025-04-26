package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

class Population_initialize_0_1_Test {

    @Mock
    private GAEnumAllelesSet allelesSet;

    private Population population;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.initMocks(this);
        population = new Population();
    }

    @Test
    void testInitialize() {
        population.initialize(allelesSet);
        // Invoke the size method on allelesSet
        when(allelesSet.size()).thenReturn(population.genoms.size());
        // Verify that the genoms list is now empty
        assertEquals(0, population.genoms.size());
    }
}

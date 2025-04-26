package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class Population_initialize_0_1_Test {

    @Test
    void initialize() {
        // Create a new Population object
        Population population = new Population();
        // Call the initialize method with a specific set of alleles
        GAEnumAllelesSet allelesSet = new GAEnumAllelesSet();
        population.initialize(allelesSet);
        // Check if the population was initialized correctly
        assertTrue(population.getSize() > 0);
        assertTrue(population.get(0).getScore() > 0);
    }
}

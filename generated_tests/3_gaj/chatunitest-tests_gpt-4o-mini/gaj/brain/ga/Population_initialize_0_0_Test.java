package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class Population_initialize_0_0_Test {

    private Population population;

    private GAEnumAllelesSet allelesSet;

    @BeforeEach
    void setUp() {
        population = new Population();
        allelesSet = Mockito.mock(GAEnumAllelesSet.class);
    }

    @Test
    void testInitializeWithValidAllelesSet() {
        // Assuming GAEnumAllelesSet has a method to get alleles
        // Mock behavior for allelesSet
        // Mockito.when(allelesSet.getAlleles()).thenReturn(Arrays.asList("A", "T", "C", "G"));
        population.initialize(allelesSet);
        // Verify that the genoms list is populated
        assertNotNull(population.genoms);
        assertFalse(population.genoms.isEmpty());
        // Additional assertions can be added based on the expected state of genoms
    }

    @Test
    void testInitializeWithNullAllelesSet() {
        Exception exception = assertThrows(NullPointerException.class, () -> {
            population.initialize(null);
        });
        // Change this based on actual implementation
        String expectedMessage = "allelesSet cannot be null";
        String actualMessage = exception.getMessage();
        assertTrue(actualMessage.contains(expectedMessage));
    }

    @Test
    void testInitializeWithEmptyAllelesSet() {
        // Mock behavior for empty alleles set
        // Mockito.when(allelesSet.getAlleles()).thenReturn(Collections.emptyList());
        population.initialize(allelesSet);
        assertNotNull(population.genoms);
        assertTrue(population.genoms.isEmpty());
    }
}

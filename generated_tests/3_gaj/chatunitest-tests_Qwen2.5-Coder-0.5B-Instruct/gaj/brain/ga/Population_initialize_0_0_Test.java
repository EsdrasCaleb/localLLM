package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class Population_initialize_0_0_Test {

    @Test
    public void testInitialize_GAEnumAllelesSet() {
        // Arrange
        GAEnumAllelesSet allelesSet = mock(GAEnumAllelesSet.class);
        Population population = new Population();
        // Act
        population.initialize(allelesSet);
        // Assert
        assertEquals(4, population.getGenomeSize());
        assertEquals(3, population.getSize());
    }
}

// Test method
package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
class VectorGenome_getGene_2_2_Test {

    @Mock
    private Evaluator evaluator;

    @InjectMocks
    private VectorGenome vectorGenome;

    @Test
    @DisplayName("Test getGene() with valid index")
    void testGetGene() {
        // Arrange
        VectorGenome vectorGenome = new VectorGenome();
        vectorGenome.setGene(0, "Gene 1");
        vectorGenome.setGene(1, "Gene 2");
        // Act
        Object gene = vectorGenome.getGene(0);
        // Assert
        assertEquals("Gene 1", gene);
    }

    @Test
    @DisplayName("Test getGene() with invalid index")
    void testGetGene_InvalidIndex() {
        // Arrange
        VectorGenome vectorGenome = new VectorGenome();
        // Act and Assert
        assertThrows(IndexOutOfBoundsException.class, () -> vectorGenome.getGene(2));
    }
}

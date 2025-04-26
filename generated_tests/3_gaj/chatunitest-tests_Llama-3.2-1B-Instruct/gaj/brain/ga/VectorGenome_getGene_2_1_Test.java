package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
public class VectorGenome_getGene_2_1_Test {

    @Mock
    private Vector genes;

    @InjectMocks
    private VectorGenome focal;

    @Test
    public void testGetGene() {
        // Arrange
        int i = 0;
        Object gene = "testGene";
        // Act
        Object result = focal.getGene(i);
        // Assert
        assertNotNull(result);
        assertEquals(gene, focal.getGene(i));
    }

    @Test
    public void testGetGene_WhenNoGene_thenReturnNull() {
        // Arrange
        int i = 0;
        Object gene = null;
        // Act
        Object result = focal.getGene(i);
        // Assert
        assertNotNull(result);
        assertEquals(gene, focal.getGene(i));
    }

    @Test
    public void testGetGene_WhenInvalidIndex_thenReturnNull() {
        // Arrange
        int i = 0;
        Object gene = "testGene";
        // Act
        Object result = focal.getGene(i);
        // Assert
        assertNotNull(result);
        assertEquals(gene, focal.getGene(i));
    }
}

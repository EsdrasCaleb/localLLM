package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class VectorGenome_getGene_2_1_Test {

    @Mock
    private VectorGenome vectorGenome;

    @BeforeEach
    public void setup() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testGetGene() {
        // Arrange
        int index = 0;
        Object gene = "testGene";
        Mockito.when(vectorGenome.getGene(index)).thenReturn(gene);
        // Act
        Object result = vectorGenome.getGene(index);
        // Assert
        assertEquals(gene, result);
    }
}

package brain.ga;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class VectorGenome_getGene_2_0_Test {

    @Mock
    private Vector<Object> genes;

    @InjectMocks
    private VectorGenome vectorGenome;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Initialize the genes vector with some mock data
        when(genes.get(0)).thenReturn("Gene0");
        when(genes.get(1)).thenReturn("Gene1");
        when(genes.size()).thenReturn(2);
        // Set the private genes field in VectorGenome
        Field genesField = VectorGenome.class.getDeclaredField("genes");
        genesField.setAccessible(true);
        genesField.set(vectorGenome, genes);
    }

    @Test
    public void testGetGene() {
        // Test getting gene at index 0
        assertEquals("Gene0", vectorGenome.getGene(0));
        // Test getting gene at index 1
        assertEquals("Gene1", vectorGenome.getGene(1));
    }

    @Test
    public void testGetGeneWithIndexOutOfBounds() {
        // Test getting gene with an index out of bounds
        Exception exception = assertThrows(IndexOutOfBoundsException.class, () -> {
            vectorGenome.getGene(2);
        });
        assertNotNull(exception);
    }
}

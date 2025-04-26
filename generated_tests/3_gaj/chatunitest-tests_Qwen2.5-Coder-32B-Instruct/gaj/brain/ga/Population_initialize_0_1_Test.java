package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
class Population_initialize_0_1_Test {

    @Mock
    private Selector selector;

    @Mock
    private Evaluator evaluator;

    @Mock
    private GAEnumAllelesSet allelesSet;

    @InjectMocks
    private Population population;

    @BeforeEach
    void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        Field genomsField = Population.class.getDeclaredField("genoms");
        genomsField.setAccessible(true);
        genomsField.set(population, new ArrayList<>());
    }

    @Test
    void testInitializeWithNullAllelesSet() throws Exception {
        // Act
        population.initialize(null);
        // Assert
        List<Genome> genoms = (List<Genome>) Population.class.getDeclaredField("genoms").get(population);
        assertTrue(genoms.isEmpty(), "Genoms list should remain empty with a null allelesSet");
    }

    @Test
    void testInitializeWithEmptyAllelesSet() throws Exception {
        // Arrange
        when(allelesSet.size()).thenReturn(0);
        // Act
        population.initialize(allelesSet);
        // Assert
        List<Genome> genoms = (List<Genome>) Population.class.getDeclaredField("genoms").get(population);
        assertTrue(genoms.isEmpty(), "Genoms list should remain empty with an empty allelesSet");
    }

    @Test
    void testInitializeWithNonEmptyAllelesSet() throws Exception {
        // Arrange
        List<Object> alleles = new ArrayList<>();
        alleles.add("A");
        alleles.add("B");
        alleles.add("C");
        alleles.add("D");
        when(allelesSet.size()).thenReturn(4);
        when(allelesSet.allele(0)).thenReturn("A");
        when(allelesSet.allele(1)).thenReturn("B");
        when(allelesSet.allele(2)).thenReturn("C");
        when(allelesSet.allele(3)).thenReturn("D");
        // Act
        population.initialize(allelesSet);
        // Assert
        List<Genome> genoms = (List<Genome>) Population.class.getDeclaredField("genoms").get(population);
        assertEquals(1, genoms.size(), "Genoms list should contain one genome with non-empty allelesSet");
        Genome genome = genoms.get(0);
        assertNotNull(genome, "Genome should not be null");
        Field genesField = Genome.class.getDeclaredField("genes");
        genesField.setAccessible(true);
        List<Object> genes = (List<Object>) genesField.get(genome);
        assertEquals(4, genes.size(), "Genome should have 4 genes");
    }
}

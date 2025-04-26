package brain.ga;

import java.lang.reflect.Field;
import java.util.Random;
import java.util.Vector;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // and returns one of them
java.util.*;

// The allele set class is a container for the different values that a gene may assume.
// If you call the allele member function with no argument,
// the allele set picks randomly from the alleles it contains
// The allele set class is a container for the different values that a gene may assume.
// If you call the allele member function with no argument,
// the allele set picks randomly from the alleles it contains
@ExtendWith(MockitoExtension.class)
public class GAEnumAllelesSet_allele_1_0_Test {

    @InjectMocks
    private GAEnumAllelesSet gaEnumAllelesSet;

    @Mock
    private Random rnd;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Initialize the alleles Vector using reflection
        Field allelesField = GAEnumAllelesSet.class.getDeclaredField("alleles");
        allelesField.setAccessible(true);
        allelesField.set(gaEnumAllelesSet, new Vector<>());
    }

    @Test
    public void testAllele_Success() throws Exception {
        // Arrange
        Vector<Object> testAlleles = new Vector<>();
        testAlleles.add("Allele1");
        testAlleles.add("Allele2");
        testAlleles.add("Allele3");
        Field allelesField = GAEnumAllelesSet.class.getDeclaredField("alleles");
        allelesField.setAccessible(true);
        allelesField.set(gaEnumAllelesSet, testAlleles);
        // Act & Assert
        assertEquals("Allele1", gaEnumAllelesSet.allele(0));
        assertEquals("Allele2", gaEnumAllelesSet.allele(1));
        assertEquals("Allele3", gaEnumAllelesSet.allele(2));
    }

    @Test
    public void testAllele_IndexOutOfBoundsException() {
        // Arrange
        Vector<Object> testAlleles = new Vector<>();
        testAlleles.add("Allele1");
        try {
            Field allelesField = GAEnumAllelesSet.class.getDeclaredField("alleles");
            allelesField.setAccessible(true);
            allelesField.set(gaEnumAllelesSet, testAlleles);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Reflection setup failed: " + e.getMessage());
        }
        // Act & Assert
        assertThrows(IndexOutOfBoundsException.class, () -> {
            gaEnumAllelesSet.allele(1);
        });
    }
}

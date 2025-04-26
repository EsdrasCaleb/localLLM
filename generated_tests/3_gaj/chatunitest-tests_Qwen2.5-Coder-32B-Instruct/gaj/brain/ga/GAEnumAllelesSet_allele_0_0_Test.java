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
public class GAEnumAllelesSet_allele_0_0_Test {

    @Mock
    private Random rnd;

    private GAEnumAllelesSet gaEnumAllelesSet;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        gaEnumAllelesSet = new GAEnumAllelesSet();
        // Set the mock Random object to the private rnd field of GAEnumAllelesSet
        Field rndField = GAEnumAllelesSet.class.getDeclaredField("rnd");
        rndField.setAccessible(true);
        rndField.set(gaEnumAllelesSet, rnd);
        // Initialize alleles with some values
        Vector<Object> alleles = new Vector<>();
        alleles.add("Allele1");
        alleles.add("Allele2");
        alleles.add("Allele3");
        gaEnumAllelesSet.setAlleles(alleles);
    }

    @Test
    public void testAlleleRandomSelection() {
        // Mock the nextInt method to return specific values to test different branches
        when(rnd.nextInt(3)).thenReturn(0, 1, 2);
        // Test if the correct alleles are returned based on the mocked random values
        assertEquals("Allele1", gaEnumAllelesSet.allele());
        assertEquals("Allele2", gaEnumAllelesSet.allele());
        assertEquals("Allele3", gaEnumAllelesSet.allele());
    }

    @Test
    public void testAllele() {
        // Test with different random indices
        when(rnd.nextInt(3)).thenReturn(0);
        assertEquals("Allele1", gaEnumAllelesSet.allele());
        when(rnd.nextInt(3)).thenReturn(1);
        assertEquals("Allele2", gaEnumAllelesSet.allele());
        when(rnd.nextInt(3)).thenReturn(2);
        assertEquals("Allele3", gaEnumAllelesSet.allele());
    }

    @Test
    public void testAlleleWithEmptyAlleles() throws Exception {
        // Set alleles to an empty vector
        Vector<Object> emptyAlleles = new Vector<>();
        gaEnumAllelesSet.setAlleles(emptyAlleles);
        // Invoke the allele() method and expect an IndexOutOfBoundsException
        Exception exception = assertThrows(IndexOutOfBoundsException.class, () -> {
            gaEnumAllelesSet.allele();
        });
        // Verify that the exception message is as expected
        assertEquals("Index 0 out of bounds for length 0", exception.getMessage());
    }
}

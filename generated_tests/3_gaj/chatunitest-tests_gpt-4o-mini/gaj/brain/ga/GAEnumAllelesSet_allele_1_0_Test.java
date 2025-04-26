package brain.ga;

import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // and returns one of them
java.util.*;
// The allele set class is a container for the different values that a gene may assume.
// If you call the allele member function with no argument,
// the allele set picks randomly from the alleles it contains

public class GAEnumAllelesSet_allele_1_0_Test {

    private GAEnumAllelesSet gaEnumAllelesSet;

    @BeforeEach
    public void setUp() {
        gaEnumAllelesSet = new GAEnumAllelesSet();
    }

    @Test
    public void testAlleleValidIndex() {
        Vector<Object> alleles = new Vector<>();
        alleles.add("A");
        alleles.add("C");
        alleles.add("G");
        alleles.add("T");
        gaEnumAllelesSet.setAlleles(alleles);
        assertEquals("A", gaEnumAllelesSet.allele(0));
        assertEquals("C", gaEnumAllelesSet.allele(1));
        assertEquals("G", gaEnumAllelesSet.allele(2));
        assertEquals("T", gaEnumAllelesSet.allele(3));
    }

    @Test
    public void testAlleleIndexOutOfBounds() {
        Vector<Object> alleles = new Vector<>();
        alleles.add("A");
        gaEnumAllelesSet.setAlleles(alleles);
        assertThrows(IndexOutOfBoundsException.class, () -> {
            // Index 1 is out of bounds
            gaEnumAllelesSet.allele(1);
        });
    }

    @Test
    public void testAlleleNegativeIndex() {
        Vector<Object> alleles = new Vector<>();
        alleles.add("A");
        gaEnumAllelesSet.setAlleles(alleles);
        assertThrows(IndexOutOfBoundsException.class, () -> {
            // Negative index is invalid
            gaEnumAllelesSet.allele(-1);
        });
    }
}

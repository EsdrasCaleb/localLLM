package brain.ga;

import java.util.Arrays;
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

public class GAEnumAllelesSet_allele_0_0_Test {

    @Test
    void testAllele() {
        GAEnumAllelesSet GA = new GAEnumAllelesSet();
        Vector alleles = new Vector();
        Vector newAlleles = new Vector(Arrays.asList(1, 2, 3, 4, 5));
        GA.setAlleles(newAlleles);
        assertEquals(newAlleles.get(0), GA.allele());
        assertEquals(newAlleles.get(1), GA.allele());
        assertEquals(newAlleles.get(2), GA.allele());
        assertEquals(newAlleles.get(3), GA.allele());
        assertEquals(newAlleles.get(4), GA.allele());
        assertEquals(newAlleles.get(5), GA.allele());
    }
}

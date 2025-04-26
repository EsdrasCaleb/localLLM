package brain.ga;

import java.util.Vector;
import java.util.Arrays;
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

    @Test
    public void testAllele() {
        GAEnumAllelesSet set = new GAEnumAllelesSet();
        Vector<String> alleles = new Vector<>(Arrays.asList("a", "b", "c"));
        set.setAlleles(alleles);
        assertEquals("a", set.allele(0));
        assertEquals("b", set.allele(1));
        assertEquals("c", set.allele(2));
    }
}

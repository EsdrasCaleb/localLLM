package brain.ga;

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
    void testAllele() {
        GAEnumAllelesSet target = new GAEnumAllelesSet();
        Vector alleles = new Vector();
        alleles.add("A");
        alleles.add("B");
        target.setAlleles(alleles);
        int i = 0;
        Object result = target.allele(i);
        assertEquals("A", result);
    }
}

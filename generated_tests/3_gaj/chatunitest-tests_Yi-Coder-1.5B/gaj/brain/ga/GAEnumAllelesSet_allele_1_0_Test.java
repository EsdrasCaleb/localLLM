package brain.ga;

import java.util.Vector;
import java.util.Random;
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

    private Vector alleles = new Vector();

    @Test
    public void testAllele() {
        alleles.add("A");
        alleles.add("T");
        alleles.add("C");
        alleles.add("G");
        GAEnumAllelesSet set = new GAEnumAllelesSet();
        set.setAlleles(alleles);
        Object allele = set.allele(0);
        assertEquals("A", allele);
        allele = set.allele(1);
        assertEquals("T", allele);
        allele = set.allele(2);
        assertEquals("C", allele);
        allele = set.allele(3);
        assertEquals("G", allele);
    }
}

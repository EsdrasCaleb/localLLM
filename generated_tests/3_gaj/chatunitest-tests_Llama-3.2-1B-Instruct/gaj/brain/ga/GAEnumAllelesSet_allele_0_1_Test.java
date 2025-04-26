package brain.ga;

import java.util.Random;
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
public class GAEnumAllelesSet_allele_0_1_Test {

    @Test
    public void testAllele() {
        GAEnumAllelesSet focal = new GAEnumAllelesSet();
        Vector newAlleles = new Vector();
        newAlleles.add("A");
        newAlleles.add("B");
        newAlleles.add("C");
        focal.setAlleles(newAlleles);
        Object result = focal.allele();
        assertEquals("A", result);
    }

    @Test
    public void testAlleleRandom() {
        GAEnumAllelesSet focal = new GAEnumAllelesSet();
        Vector newAlleles = new Vector();
        newAlleles.add("A");
        newAlleles.add("B");
        newAlleles.add("C");
        focal.setAlleles(newAlleles);
        Object result = focal.allele();
        assertEquals("B", result);
    }

    @Test
    public void testAlleleZero() {
        GAEnumAllelesSet focal = new GAEnumAllelesSet();
        Vector newAlleles = new Vector();
        newAlleles.add("A");
        newAlleles.add("B");
        newAlleles.add("C");
        focal.setAlleles(newAlleles);
        Object result = focal.allele();
        assertEquals(0, focal.allele());
    }
}

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
public class GAEnumAllelesSet_size_3_0_Test {

    @Test
    public void testSize() {
        GAEnumAllelesSet allelesSet = new GAEnumAllelesSet();
        allelesSet.setAlleles(new Vector());
        assertEquals(0, allelesSet.size());
        // Add some alleles
        Vector newAlleles = new Vector();
        newAlleles.add("A");
        newAlleles.add("B");
        allelesSet.setAlleles(newAlleles);
        assertEquals(2, allelesSet.size());
    }
}

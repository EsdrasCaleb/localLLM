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

class GAEnumAllelesSet_size_3_4_Test {

    @Test
    void testSize() {
        GAEnumAllelesSet GA = new GAEnumAllelesSet();
        GA.setAlleles(new Vector());
        // Test with empty vector
        assertEquals(0, GA.size());
        // Test with non-empty vector
        GA.setAlleles(new Vector(10));
        assertEquals(10, GA.size());
        // Test with vector of different sizes
        GA.setAlleles(new Vector(5));
        assertEquals(5, GA.size());
        // Test with vector of different capacity
        GA.setAlleles(new Vector(20));
        assertEquals(20, GA.size());
    }
}

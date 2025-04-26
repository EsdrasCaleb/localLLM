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

class GAEnumAllelesSet_size_3_1_Test {

    @Test
    public void testSize() throws Exception {
        // Create an instance of GAEnumAllelesSet
        GAEnumAllelesSet gaEnumAllelesSet = new GAEnumAllelesSet();
        // Create a Vector object with some initial elements
        Vector<String> initialElements = new Vector<>();
        initialElements.add("A");
        initialElements.add("B");
        initialElements.add("C");
        // Set the initial elements to the GAEnumAllelesSet instance
        gaEnumAllelesSet.setAlleles(initialElements);
        // Get the size of the alleles vector
        int size = gaEnumAllelesSet.size();
        // Assert that the size is as expected
        assertEquals(3, size);
    }
}

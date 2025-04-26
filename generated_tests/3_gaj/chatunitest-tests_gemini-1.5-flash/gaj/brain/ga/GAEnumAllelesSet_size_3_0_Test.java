package brain.ga;

import java.lang.reflect.Field;
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
public class GAEnumAllelesSet_size_3_0_Test {

    private GAEnumAllelesSet gaEnumAllelesSet;

    @BeforeEach
    void setUp() {
        gaEnumAllelesSet = new GAEnumAllelesSet();
    }

    @Test
    void testSizeNotEmpty() {
        Vector<Object> alleles = new Vector<>();
        alleles.add(new Object());
        alleles.add(new Object());
        alleles.add(new Object());
        try {
            Field field = GAEnumAllelesSet.class.getDeclaredField("alleles");
            field.setAccessible(true);
            field.set(gaEnumAllelesSet, alleles);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set alleles field: " + e.getMessage());
        }
        assertEquals(3, gaEnumAllelesSet.size());
    }
}

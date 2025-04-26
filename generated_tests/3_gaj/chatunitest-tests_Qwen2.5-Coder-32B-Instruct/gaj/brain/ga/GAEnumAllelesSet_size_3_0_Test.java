package brain.ga;

import java.lang.reflect.Field;
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

    @InjectMocks
    private GAEnumAllelesSet gaEnumAllelesSet;

    @Mock
    private Vector<Object> alleles;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Injecting the mock vector into the private field 'alleles'
        Field field = GAEnumAllelesSet.class.getDeclaredField("alleles");
        field.setAccessible(true);
        field.set(gaEnumAllelesSet, alleles);
    }

    @Test
    public void testSize_EmptyVector() {
        when(alleles.size()).thenReturn(0);
        assertEquals(0, gaEnumAllelesSet.size());
    }

    @Test
    public void testSize_NonEmptyVector() {
        when(alleles.size()).thenReturn(5);
        assertEquals(5, gaEnumAllelesSet.size());
    }
}

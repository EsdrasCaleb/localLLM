package brain.ga;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class Population_initialize_0_0_Test {

    @Mock
    private GAEnumAllelesSet allelesSet;

    @InjectMocks
    private Population population;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testInitialize() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        List<Genome> genoms = new ArrayList<>();
        Field genomsField = Population.class.getDeclaredField("genoms");
        genomsField.setAccessible(true);
        genomsField.set(population, genoms);
        // Act
        population.initialize(allelesSet);
        // Assert
        assertNotNull(genoms);
        // Assuming initialize should clear the genoms list
        assertEquals(0, genoms.size());
    }
}

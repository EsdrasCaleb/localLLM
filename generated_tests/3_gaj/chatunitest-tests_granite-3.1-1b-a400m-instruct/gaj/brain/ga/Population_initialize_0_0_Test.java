package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class Population_initialize_0_0_Test {

    @Test
    public void testInitialize() {
        Population population = new Population();
        population.initialize(new GAEnumAllelesSet());
        assertEquals(0, population.getSize());
    }
}

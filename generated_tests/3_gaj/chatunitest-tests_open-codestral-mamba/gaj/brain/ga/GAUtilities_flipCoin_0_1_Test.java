package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
class GAUtilities_flipCoin_0_1_Test {

    @Mock
    private Random mockRandom;

    @Test
    void testFlipCoin() {
        GAUtilities gaUtilities = new GAUtilities();
        // Mocking the Random class to always return 0.0
        try {
            Field rndField = GAUtilities.class.getDeclaredField("rnd");
            rndField.setAccessible(true);
            rndField.set(gaUtilities, mockRandom);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error while mocking Random");
        }
        // Testing the flipCoin method
        when(mockRandom.nextBoolean()).thenReturn(false);
        assertFalse(gaUtilities.flipCoin(0.5));
    }
}

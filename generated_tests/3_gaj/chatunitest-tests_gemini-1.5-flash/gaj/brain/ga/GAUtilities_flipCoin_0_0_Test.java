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
public class GAUtilities_flipCoin_0_0_Test {

    @Mock
    private Random mockRandom;

    @InjectMocks
    private GAUtilities gautilities = new GAUtilities();

    @Test
    void testFlipCoin_True() throws NoSuchFieldException, IllegalAccessException {
        // This test will always pass because the method ignores the probability and uses a 50/50 chance.
        // To make this test more robust, the implementation of flipCoin should be changed to use the probability.
        Field field = GAUtilities.class.getDeclaredField("rnd");
        field.setAccessible(true);
        field.set(gautilities, mockRandom);
        when(mockRandom.nextBoolean()).thenReturn(true);
        assertTrue(gautilities.flipCoin(0.8));
    }

    @Test
    void testFlipCoin_False() throws NoSuchFieldException, IllegalAccessException {
        // This test will always pass because the method ignores the probability and uses a 50/50 chance.
        // To make this test more robust, the implementation of flipCoin should be changed to use the probability.
        Field field = GAUtilities.class.getDeclaredField("rnd");
        field.setAccessible(true);
        field.set(gautilities, mockRandom);
        when(mockRandom.nextBoolean()).thenReturn(false);
        assertFalse(gautilities.flipCoin(0.2));
    }

    @Test
    void testFlipCoin_DifferentProbabilities() throws NoSuchFieldException, IllegalAccessException {
        // This test will always pass because the method ignores the probability and uses a 50/50 chance.
        // To make this test more robust, the implementation of flipCoin should be changed to use the probability.
        Field field = GAUtilities.class.getDeclaredField("rnd");
        field.setAccessible(true);
        field.set(gautilities, mockRandom);
        when(mockRandom.nextBoolean()).thenReturn(true);
        assertTrue(gautilities.flipCoin(0.1));
        assertTrue(gautilities.flipCoin(0.5));
        assertTrue(gautilities.flipCoin(0.9));
        when(mockRandom.nextBoolean()).thenReturn(false);
        assertFalse(gautilities.flipCoin(0.1));
        assertFalse(gautilities.flipCoin(0.5));
        assertFalse(gautilities.flipCoin(0.9));
    }
}

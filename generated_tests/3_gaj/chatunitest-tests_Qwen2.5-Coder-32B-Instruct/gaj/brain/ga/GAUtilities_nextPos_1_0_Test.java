package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
public class GAUtilities_nextPos_1_0_Test {

    @Mock
    private Random mockRandom;

    @InjectMocks
    private GAUtilities gaUtilities;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        Field rndField = GAUtilities.class.getDeclaredField("rnd");
        rndField.setAccessible(true);
        rndField.set(gaUtilities, mockRandom);
    }

    @Test
    public void testNextPos() throws Exception {
        // Arrange
        int size = 10;
        int expectedPos = 5;
        when(mockRandom.nextInt(size)).thenReturn(expectedPos);
        // Act
        int result = invokePrivateNextPos(size);
        // Assert
        assertEquals(expectedPos, result);
        verify(mockRandom, times(1)).nextInt(size);
    }

    private int invokePrivateNextPos(int size) throws Exception {
        Method method = GAUtilities.class.getDeclaredMethod("nextPos", int.class);
        method.setAccessible(true);
        return (int) method.invoke(gaUtilities, size);
    }

    @Test
    public void testNextPosWithN1() {
        when(mockRandom.nextInt(anyInt())).thenReturn(0);
        assertEquals(0, GAUtilities.nextPos(1));
    }

    @Test
    public void testNextPosWithN2() {
        when(mockRandom.nextInt(anyInt())).thenReturn(0);
        assertEquals(0, GAUtilities.nextPos(2));
        when(mockRandom.nextInt(anyInt())).thenReturn(1);
        assertEquals(1, GAUtilities.nextPos(2));
        when(mockRandom.nextInt(anyInt())).thenReturn(2);
        assertEquals(1, GAUtilities.nextPos(2));
    }

    @Test
    public void testNextPosWithN3() {
        when(mockRandom.nextInt(anyInt())).thenReturn(0);
        assertEquals(0, GAUtilities.nextPos(3));
        when(mockRandom.nextInt(anyInt())).thenReturn(1);
        assertEquals(1, GAUtilities.nextPos(3));
        when(mockRandom.nextInt(anyInt())).thenReturn(2);
        assertEquals(1, GAUtilities.nextPos(3));
        when(mockRandom.nextInt(anyInt())).thenReturn(3);
        assertEquals(2, GAUtilities.nextPos(3));
        when(mockRandom.nextInt(anyInt())).thenReturn(4);
        assertEquals(2, GAUtilities.nextPos(3));
        when(mockRandom.nextInt(anyInt())).thenReturn(5);
        assertEquals(2, GAUtilities.nextPos(3));
    }

    @Test
    public void testNextPosWithN5() {
        when(mockRandom.nextInt(anyInt())).thenReturn(0);
        assertEquals(0, GAUtilities.nextPos(5));
        when(mockRandom.nextInt(anyInt())).thenReturn(1);
        assertEquals(1, GAUtilities.nextPos(5));
        when(mockRandom.nextInt(anyInt())).thenReturn(2);
        assertEquals(1, GAUtilities.nextPos(5));
        when(mockRandom.nextInt(anyInt())).thenReturn(3);
        assertEquals(2, GAUtilities.nextPos(5));
        when(mockRandom.nextInt(anyInt())).thenReturn(4);
        assertEquals(2, GAUtilities.nextPos(5));
        when(mockRandom.nextInt(anyInt())).thenReturn(5);
        assertEquals(2, GAUtilities.nextPos(5));
        when(mockRandom.nextInt(anyInt())).thenReturn(6);
        assertEquals(3, GAUtilities.nextPos(5));
        when(mockRandom.nextInt(anyInt())).thenReturn(7);
        assertEquals(3, GAUtilities.nextPos(5));
        when(mockRandom.nextInt(anyInt())).thenReturn(8);
        assertEquals(3, GAUtilities.nextPos(5));
        when(mockRandom.nextInt(anyInt())).thenReturn(9);
        assertEquals(3, GAUtilities.nextPos(5));
        when(mockRandom.nextInt(anyInt())).thenReturn(10);
        assertEquals(4, GAUtilities.nextPos(5));
        when(mockRandom.nextInt(anyInt())).thenReturn(11);
        assertEquals(4, GAUtilities.nextPos(5));
        when(mockRandom.nextInt(anyInt())).thenReturn(12);
        assertEquals(4, GAUtilities.nextPos(5));
        when(mockRandom.nextInt(anyInt())).thenReturn(13);
        assertEquals(4, GAUtilities.nextPos(5));
        when(mockRandom.nextInt(anyInt())).thenReturn(14);
        assertEquals(4, GAUtilities.nextPos(5));
        when(mockRandom.nextInt(anyInt())).thenReturn(15);
        assertEquals(4, GAUtilities.nextPos(5));
    }
}

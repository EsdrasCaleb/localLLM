package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Directors_toString_4_0_Test {

    @Mock
    private ArrayList directors;

    @InjectMocks
    private Directors directorsUnderTest;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testToString() {
        when(directors.size()).thenReturn(2);
        when(directors.get(0)).thenReturn("Director 1");
        when(directors.get(1)).thenReturn("Director 2");
        String expected = "# of Directors = 2\n" + "Director - Director 1\n" + "Director - Director 2\n";
        String actual = directorsUnderTest.toString();
        assertEquals(expected, actual);
    }
}

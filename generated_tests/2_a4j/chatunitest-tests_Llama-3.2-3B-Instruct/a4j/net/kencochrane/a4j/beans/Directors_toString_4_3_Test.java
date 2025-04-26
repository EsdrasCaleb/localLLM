// Directors_toString_4_3_Test.java
package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
@MockitoSettings
public class Directors_toString_4_3_Test {

    @InjectMocks
    private Directors directors;

    @BeforeEach
    public void setup() {
        directors = new Directors();
    }

    @Test
    public void testToString_EmptyList_ReturnsDefaultMessage() {
        String result = directors.toString();
        assertEquals("Director is null or size 0", result);
    }

    @Test
    public void testToString_NonEmptyList_ReturnsCorrectString() {
        directors.setDirector(new String[] { "Director1", "Director2" });
        String result = directors.toString();
        assertEquals("Director - Director1\nDirector - Director2\n# of Directors = 2", result);
    }

    @Test
    public void testToString_NullList_ThrowsNullPointerException() {
        assertThrows(NullPointerException.class, () -> directors.toString());
    }
}

package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Directors_getDirector_3_0_Test {

    @Test
    public void testGetDirector_InValidIndex_ReturnsNull() {
        Directors directors = new Directors();
        String result = directors.getDirector(10);
        assertEquals(null, result);
    }

    @Test
    public void testGetDirector_ValidIndex_ReturnsDirector() {
        Directors directors = new Directors();
        directors.setDirector(new String[] { "John", "Mary", "David" });
        String result = directors.getDirector(1);
        assertEquals("Mary", result);
    }

    @Test
    public void testGetDirector_EmptyList_ReturnsNull() {
        Directors directors = new Directors();
        String result = directors.getDirector(0);
        assertEquals(null, result);
    }

    @Test
    public void testGetDirector_NullList_ThrowsNullPointerException() {
        Directors directors = null;
        assertThrows(NullPointerException.class, () -> directors.getDirector(0));
    }
}

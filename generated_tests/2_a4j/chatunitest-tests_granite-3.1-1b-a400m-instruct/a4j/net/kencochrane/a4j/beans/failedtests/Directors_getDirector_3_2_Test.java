package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Directors_getDirector_3_2_Test {

    @Test
    public void testGetDirector_ValidIndex() {
        Directors director = new Directors();
        director.setDirector(new String[] { "Tom", "Hanks", "Tom Hanks" });
        assertEquals(new String[] { "Tom", "Hanks" }, director.getDirector(1));
    }

    @Test
    public void testGetDirector_OutOfBoundsIndex() {
        Directors director = new Directors();
        director.setDirector(new String[] { "Tom", "Hanks" });
        assertNull(director.getDirector(5));
    }

    @Test
    public void testGetDirector_NegativeIndex() {
        Directors director = new Directors();
        director.setDirector(new String[] { "Tom", "Hanks" });
        assertNull(director.getDirector(-1));
    }
}

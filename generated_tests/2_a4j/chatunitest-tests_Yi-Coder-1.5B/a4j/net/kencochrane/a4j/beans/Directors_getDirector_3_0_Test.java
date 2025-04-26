package net.kencochrane.a4j.beans;

import static org.junit.Assert.assertArrayEquals;
import org.junit.Before;
import org.junit.Test;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

/**
 * Unit test for simple App.
 */
public class Directors_getDirector_3_0_Test {

    private Directors directors;

    @Before
    public void setUp() throws Exception {
        directors = new Directors();
    }

    @Test
    public void testDirector_0() {
        directors.setDirector(new String[] { "Director 1" });
        String[] expected = new String[] { "Director 1" };
        String[] actual = directors.getDirector();
        assertArrayEquals(expected, actual);
    }

    @Test
    public void testDirector_1() {
        directors.setDirector(new String[] { "Director 1", "Director 2" });
        String[] expected = new String[] { "Director 1", "Director 2" };
        String[] actual = directors.getDirector();
        assertArrayEquals(expected, actual);
    }

    @Test
    public void testDirector_2() {
        directors.setDirector(new String[] { "Director 1", "Director 2", "Director 3" });
        String[] expected = new String[] { "Director 1", "Director 2", "Director 3" };
        String[] actual = directors.getDirector();
        assertArrayEquals(expected, actual);
    }

    @Test
    public void testDirector_3() {
        directors.setDirector(new String[] { "Director 1", "Director 2", "Director 3", "Director 4" });
        String[] expected = new String[] { "Director 1", "Director 2", "Director 3", "Director 4" };
        String[] actual = directors.getDirector();
        assertArrayEquals(expected, actual);
    }

    @Test
    public void testDirector_4() {
        directors.setDirector(new String[] { "Director 1", "Director 2", "Director 3", "Director 4", "Director 5" });
        String[] expected = new String[] { "Director 1", "Director 2", "Director 3", "Director 4", "Director 5" };
        String[] actual = directors.getDirector();
        assertArrayEquals(expected, actual);
    }
}

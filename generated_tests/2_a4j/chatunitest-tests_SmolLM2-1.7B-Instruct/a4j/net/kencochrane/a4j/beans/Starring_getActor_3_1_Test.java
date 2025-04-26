package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Starring_getActor_3_1_Test {

    @Test
    public void testGetActorValidIndex() {
        Starring star = new Starring();
        star.setActor(new String[] { "Tom", "Dick", "Harry" });
        String expected = "Tom";
        String actual = star.getActor(0);
        assertEquals(expected, actual);
    }

    @Test
    public void testGetActorInvalidIndex() {
        Starring star = new Starring();
        star.setActor(new String[] { "Tom", "Dick", "Harry" });
        String expected = "Dick";
        String actual = star.getActor(1);
        assertEquals(expected, actual);
    }

    @Test
    public void testGetActorOutOfRange() {
        Starring star = new Starring();
        star.setActor(new String[] { "Tom", "Dick", "Harry" });
        String expected = "Harry";
        String actual = star.getActor(2);
        assertEquals(expected, actual);
    }
}

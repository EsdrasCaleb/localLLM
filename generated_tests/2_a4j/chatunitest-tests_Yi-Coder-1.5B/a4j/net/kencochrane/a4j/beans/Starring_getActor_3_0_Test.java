package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class Starring_getActor_3_0_Test {

    // Test class
    @Test
    public void testGetActor() {
        Starring star = new Starring();
        star.setActor(new String[] { "John", "Jane", "Joe" });
        assertEquals("Jane", star.getActor(1));
        assertEquals("John", star.getActor(0));
        assertEquals("Joe", star.getActor(2));
        assertEquals(null, star.getActor(3));
    }
}

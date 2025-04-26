package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Starring_toString_4_0_Test {

    @Test
    public void testToString() throws Exception {
        Starring starring = new Starring();
        ArrayList<String> actors = new ArrayList<>();
        actors.add("Actor1");
        actors.add("Actor2");
        starring.setActor(actors.toArray(new String[0]));
        String expectedOutput = "# of Actors = 2\n" + "Actor - Actor1\n" + "Actor - Actor2\n";
        Field field = Starring.class.getDeclaredField("actors");
        field.setAccessible(true);
        String result = starring.toString();
        assertEquals(expectedOutput, result);
    }
}

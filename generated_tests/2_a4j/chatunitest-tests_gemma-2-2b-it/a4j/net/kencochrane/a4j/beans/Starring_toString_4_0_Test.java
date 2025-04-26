package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Starring_toString_4_0_Test {

    @Test
    void testToString() {
        Starring starring = new Starring();
        starring.setActor(new String[] { "John", "Jane", "Bruce" });
        String output = starring.toString();
        System.out.println(output);
    }
}

package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class FullProduct_printFullProduct_8_1_Test {

    @Test
    void testPrintFullProduct_empty() {
        FullProduct fp = new FullProduct();
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream ps = new PrintStream(baos);
        PrintStream old = System.out;
        System.setOut(ps);
        fp.printFullProduct();
        System.out.flush();
        System.setOut(old);
        String expected = "ProductDetails{productName='null', productPrice=0.0}\n" + "\n" + "-- Accessories --\n" + "-- Similar Products --\n";
        assertEquals(expected, baos.toString());
    }

    @Test
    void testPrintFullProduct_withAccessories() {
        FullProduct fp = new FullProduct();
        ArrayList<String> accessories = new ArrayList<>();
        accessories.add("Accessory 1");
        accessories.add("Accessory 2");
        try {
            Field accessoriesField = FullProduct.class.getDeclaredField("accessories");
            accessoriesField.setAccessible(true);
            accessoriesField.set(fp, accessories);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
            fail("Failed to set accessories field");
        }
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream ps = new PrintStream(baos);
        PrintStream old = System.out;
        System.setOut(ps);
        fp.printFullProduct();
        System.out.flush();
        System.setOut(old);
        String expected = "ProductDetails{productName='null', productPrice=0.0}\n" + "\n" + "-- Accessories --\n" + "Accessory 1\n" + "Accessory 2\n" + "-- Similar Products --\n";
        assertEquals(expected, baos.toString());
    }

    @Test
    void testPrintFullProduct_withSimilarItems() {
        FullProduct fp = new FullProduct();
        ArrayList<String> similarItems = new ArrayList<>();
        similarItems.add("Similar 1");
        similarItems.add("Similar 2");
        try {
            Field similarItemsField = FullProduct.class.getDeclaredField("similarItems");
            similarItemsField.setAccessible(true);
            similarItemsField.set(fp, similarItems);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
            fail("Failed to set similarItems field");
        }
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream ps = new PrintStream(baos);
        PrintStream old = System.out;
        System.setOut(ps);
        fp.printFullProduct();
        System.out.flush();
        System.setOut(old);
        String expected = "ProductDetails{productName='null', productPrice=0.0}\n" + "\n" + "-- Accessories --\n" + "-- Similar Products --\n" + "Similar 1\n" + "Similar 2\n";
        assertEquals(expected, baos.toString());
    }

    @Test
    void testPrintFullProduct_withAll() {
        FullProduct fp = new FullProduct();
        ArrayList<String> accessories = new ArrayList<>();
        accessories.add("Accessory 1");
        ArrayList<String> similarItems = new ArrayList<>();
        similarItems.add("Similar 1");
        try {
            Field accessoriesField = FullProduct.class.getDeclaredField("accessories");
            accessoriesField.setAccessible(true);
            accessoriesField.set(fp, accessories);
            Field similarItemsField = FullProduct.class.getDeclaredField("similarItems");
            similarItemsField.setAccessible(true);
            similarItemsField.set(fp, similarItems);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
            fail("Failed to set fields");
        }
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream ps = new PrintStream(baos);
        PrintStream old = System.out;
        System.setOut(ps);
        fp.printFullProduct();
    }
}

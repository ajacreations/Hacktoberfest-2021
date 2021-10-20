function palindrome(str) { //Checks if the input is a Palindrome (ignoring punctuation, case, and spacing).
	let newStr;

	str = str.replace(/[^0-9a-z]/gi, ''); //Remove all non-alphanumeric characters. Only letters & numbers left.
	str = str.toLowerCase(); //Converts string to lowercase.

	newStr = str.split(''); //Splits string into array on each letter.
	newStr.reverse(); //Reverses the order of the array.
	newStr = newStr.join(''); //Converts the array back into a string.

	console.log("Old String: " + str);
	console.log("New String: " + newStr);

	if (str === newStr) { //Strings match. Palindrome.
		console.log("true \n");
		return true;
	}
	console.log("false \n"); //Strings do not match. Not a Palindrome.
	return false;
}

palindrome("eye"); //True
palindrome("_eye"); //True
palindrome("race car"); //True
palindrome("not a palindrome"); //False
palindrome("A man, a plan, a canal. Panama"); //True
palindrome("never odd or even"); //True
palindrome("nope"); //False
palindrome("almostomla"); //False
palindrome("My age is 0, 0 si ega ym."); //True
palindrome("1 eye for of 1 eye."); //False
palindrome("0_0 (: /-\ :) 0-0"); //True
palindrome("five|\_/|four"); //False